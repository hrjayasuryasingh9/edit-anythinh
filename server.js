import express from "express";

const {
  OPENROUTER_API_KEY,
  GEMINI_API_KEY,
  OPENAI_API_KEY,
  OPENAI_MODEL = "gpt-image-2.5-sunburst",
  MODEL = "google/gemini-2.5-flash-image",
  GEMINI_MODEL = "gemini-2.5-flash-image",
  PORT = 3100,
} = process.env;

if (!OPENAI_API_KEY && !OPENROUTER_API_KEY && !GEMINI_API_KEY) {
  console.error("Set OPENAI_API_KEY, GEMINI_API_KEY or OPENROUTER_API_KEY in .env (see .env.example).");
  process.exit(1);
}

const MAX_IMAGES = 8;

const app = express();
app.use(express.json({ limit: "60mb" }));
app.use(express.static("public"));

const SYSTEM_PROMPT =
  "You are an image editing engine. Carry out exactly the instruction given, no more and no less. " +
  "Change only what the instruction names; keep everything else identical: composition, faces, " +
  "identity, lighting, colors, text, framing and resolution. Do not add elements, styles, filters, " +
  "watermarks or captions that were not requested. Always respond with an image, not with text.";

// With several photos, the first is the one being edited and the rest are references to use.
function withContext(prompt, count) {
  return count > 1
    ? `The first image is the base image to edit; the other ${count - 1} are reference images. ${prompt}`
    : prompt;
}

function parseDataUrl(image) {
  const [, mime, data] = image.match(/^data:([^;]+);base64,(.*)$/s) ?? [];
  if (!data) throw new Error("Invalid image data.");
  return { mime, data };
}

function noImage(message) {
  const err = new Error(message || "The model returned no image.");
  err.noImage = true;
  return err;
}

async function callOpenAI(images, prompt) {
  const form = new FormData();
  form.append("model", OPENAI_MODEL);
  form.append("prompt", `${SYSTEM_PROMPT}\n\nInstruction: ${prompt}`);
  form.append("quality", "high");
  images.map(parseDataUrl).forEach(({ mime, data }, i) => {
    form.append("image[]", new Blob([Buffer.from(data, "base64")], { type: mime }), `image${i}.${mime.split("/")[1]}`);
  });
  const res = await fetch("https://api.openai.com/v1/images/edits", {
    method: "POST",
    headers: { Authorization: `Bearer ${OPENAI_API_KEY}` },
    body: form,
  });
  const json = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(json?.error?.message || `OpenAI error ${res.status}`);
  const b64 = json?.data?.[0]?.b64_json;
  if (!b64) throw noImage();
  return `data:image/png;base64,${b64}`;
}

async function callGemini(images, prompt) {
  const imageParts = images.map(parseDataUrl).map(({ mime, data }) => ({ inline_data: { mime_type: mime, data } }));
  const res = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent`,
    {
      method: "POST",
      headers: { "x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json" },
      body: JSON.stringify({
        systemInstruction: { parts: [{ text: SYSTEM_PROMPT }] },
        contents: [{ role: "user", parts: [{ text: prompt }, ...imageParts] }],
        generationConfig: { responseModalities: ["IMAGE", "TEXT"] },
      }),
    },
  );
  const json = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(json?.error?.message || `Gemini error ${res.status}`);
  const parts = json?.candidates?.[0]?.content?.parts ?? [];
  const img = parts.map((p) => p.inlineData ?? p.inline_data).find((d) => d?.data);
  if (!img) {
    throw noImage(
      parts.map((p) => p.text).filter(Boolean).join(" ") || json?.promptFeedback?.blockReason,
    );
  }
  return `data:${img.mimeType ?? img.mime_type ?? "image/png"};base64,${img.data}`;
}

async function callOpenRouter(images, prompt) {
  const res = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${OPENROUTER_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: MODEL,
      modalities: ["image", "text"],
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        {
          role: "user",
          content: [
            { type: "text", text: prompt },
            ...images.map((url) => ({ type: "image_url", image_url: { url } })),
          ],
        },
      ],
    }),
  });

  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data?.error?.message || `OpenRouter error ${res.status}`);
  }
  const message = data?.choices?.[0]?.message;
  const url = message?.images?.[0]?.image_url?.url;
  if (!url) throw noImage(typeof message?.content === "string" ? message.content : "");
  return url;
}

app.post("/api/edit", async (req, res) => {
  const { images, prompt } = req.body ?? {};
  if (
    !Array.isArray(images) ||
    !images.length ||
    images.length > MAX_IMAGES ||
    !images.every((i) => typeof i === "string" && i.startsWith("data:image/"))
  ) {
    return res.status(400).json({ error: `Provide 1 to ${MAX_IMAGES} images.` });
  }
  if (typeof prompt !== "string" || !prompt.trim()) {
    return res.status(400).json({ error: "Describe the edit you want." });
  }

  const edit = OPENAI_API_KEY ? callOpenAI : GEMINI_API_KEY ? callGemini : callOpenRouter;
  const fullPrompt = withContext(prompt.trim(), images.length);

  // Models occasionally reply with text instead of an image; retry before giving up.
  const attempts = 3;
  let lastError;
  for (let i = 0; i < attempts; i++) {
    try {
      return res.json({ image: await edit(images, fullPrompt) });
    } catch (err) {
      lastError = err;
      if (!err.noImage) break;
    }
  }
  res.status(502).json({ error: lastError?.message || "Edit failed." });
});

app.listen(PORT, () => console.log(`Edit Anything running at http://localhost:${PORT}`));
