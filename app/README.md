# Run the Streamlit Demo App

This folder contains the Streamlit demo application for the AI Conversation Summarizer.

## ⚡ Quick Start (No Training Required!)

The demo works **out of the box** with the base FLAN-T5 model:

```bash
# From the project root directory
streamlit run app/demo_app.py
```

Or from this directory:
```bash
cd app
streamlit run demo_app.py
```

**First run:** The app will download the FLAN-T5-base model (~1GB) from HuggingFace. This happens automatically and only once.

## 🎯 Using Base vs Fine-Tuned Model

### Base Model (Default)
- ✅ **No training required** - works immediately
- ✅ Downloads automatically (~1GB, one-time)
- ⚠️ Generic summarization (not optimized for conversations)
- ⚠️ Lower accuracy than fine-tuned version

### Fine-Tuned Model (Optional)
- ✅ Better accuracy for conversation summarization
- ✅ Trained on DialogSum dataset
- ⚠️ Requires running training first (see [main README](../README.md))

To use your fine-tuned model:
1. Train the model using the main project scripts
2. Update the `load_model()` call in `demo_app.py` with your PEFT weights path

# Run the Streamlit Demo App

This folder contains the Streamlit demo application for the AI Conversation Summarizer.

## ⚡ Quick Start (No Training Required!)

The demo works **out of the box** with the base FLAN-T5 model:

```bash
# From the project root directory
streamlit run app/demo_app.py
```

Or from this directory:
```bash
cd app
streamlit run demo_app.py
```

**First run:** The app will download the FLAN-T5-base model (~1GB) from HuggingFace. This happens automatically and only once.

## 🎯 Using Base vs Fine-Tuned Model

### Base Model (Default)
- ✅ **No training required** - works immediately
- ✅ Downloads automatically (~1GB, one-time)
- ⚠️ Generic summarization (not optimized for conversations)
- ⚠️ Lower accuracy than fine-tuned version

### Fine-Tuned Model (Optional)
- ✅ Better accuracy for conversation summarization
- ✅ Trained on DialogSum dataset
- ⚠️ Requires running training first (see [main README](../README.md))

To use your fine-tuned model:
1. Train the model using the main project scripts
2. Update the `load_model()` call in `demo_app.py` with your PEFT weights path

## What's Included

- `demo_app.py` - Main Streamlit application with three tabs:
  - **Try Demo**: Interactive conversation summarization
  - **Business Value**: ROI calculator and use cases
  - **Technical Details**: Architecture and implementation info

- `.streamlit/config.toml` - Streamlit theme and configuration

## Deployment

### Deploy to Streamlit Cloud (Free)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Set the main file path: `app/demo_app.py`
5. Deploy!

### Deploy to Other Platforms

- **Heroku**: Add a `Procfile` with `web: streamlit run app/demo_app.py`
- **AWS/GCP**: Use Docker with the base image from Streamlit
- **Azure**: Deploy as a Web App

## Customization

Update these placeholders in `demo_app.py`:
- `your.email@example.com` → your email
- `yourusername` → your GitHub username
- Demo URL links once deployed
