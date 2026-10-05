# 07 — Streamlit Demo App

**Code:** [`app.py`](../app.py)

```bash
streamlit run app.py
```

Opens in the browser at `http://localhost:8501`.

---

## 1. What the app does

1. **Model** dropdown — lists every `checkpoints/*_best.pth` found.
2. **Upload** a face image (jpg / png / webp).
3. The image is preprocessed **exactly like the validation set** (resize 224×224, ImageNet
   normalisation — via `get_transforms(train=False)`), so the model sees inputs in the same
   format it was trained on.
4. Shows the **prediction** (REAL / FAKE), the **confidence**, and P(fake) as a bar.
5. Shows the input next to its **Grad-CAM overlay** (can be turned off with the checkbox).

The interface is deliberately plain — default Streamlit widgets, no custom styling.
"Upload → predict → show result" is all the demo needs.

## 2. How Streamlit works (short)

- A Streamlit app is a normal Python script. Every widget interaction (upload, checkbox)
  **re-runs the whole script from top to bottom**.
- Widgets return their current value (`uploaded = st.file_uploader(...)`).
- Re-loading the model on every rerun would be slow, so `load_explainer` is decorated with
  **`@st.cache_resource`**: it runs once per checkpoint and the model object is reused afterwards.

## 3. Confidence vs probability

The model gives `P(fake)`. The displayed **confidence** is the probability of the *predicted*
class:

```
confidence = P(fake)       if prediction is FAKE
           = 1 − P(fake)   if prediction is REAL
```

So it is always between 50% and 100%.

⚠️ A high confidence is **not** a guarantee. Neural networks are often over-confident,
especially on inputs unlike the training data.

## 4. Testing with out-of-distribution images

Try (and note results for the report):

| Input | What to expect |
|-------|----------------|
| Test-set images | correct, very confident |
| Your own phone photo | may work; non-aligned, different camera → less reliable |
| thispersondoesnotexist.com (StyleGAN2/3) | newer generator — may or may not be detected |
| Diffusion-model faces (Midjourney, SD) | different artifacts — likely unreliable |
| Celebrity photos from the web | heavily edited/compressed → possible false "fake" |
| Non-face images | meaningless output (the model always answers real/fake) |

**Out-of-distribution (OOD)** inputs = images that differ from the training distribution
(aligned 256×256 FFHQ / StyleGAN faces). Behaviour on them is undefined — the model has never
seen anything similar. This is a key limitation to mention.

## 5. Screenshots for the presentation

Take screenshots of: a correct REAL, a correct FAKE, and one interesting failure. Save them in
`outputs/` (or the report folder) for the slides.
