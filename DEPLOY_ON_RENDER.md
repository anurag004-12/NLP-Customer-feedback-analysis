# Deployment Guide: Render

This guide walks you through deploying the AI Customer Feedback Analysis application on Render.

## Prerequisites

- GitHub repository with your code pushed
- Render account (free tier available at https://render.com)

## Deployment Steps

### 1. Prepare Your Repository

Ensure your repository contains:
- ✅ `Dockerfile` - Already configured for Streamlit
- ✅ `requirements.txt` - All Python dependencies (fixed UTF-8 encoding)
- ✅ `.streamlit/config.toml` - Streamlit configuration
- ✅ `render.yaml` - Render deployment configuration
- ✅ `runtime.txt` - Python version specification

### 2. Push Code to GitHub

```bash
git add .
git commit -m "Prepare for Render deployment"
git push origin main
```

### 3. Deploy on Render

#### Option A: Using Render Dashboard (Recommended for Beginners)

1. Go to https://dashboard.render.com
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repository:
   - Select your **NLP-Customer-feedback-analysis** repository
   - Click **Connect**
4. Fill in the deployment settings:
   - **Name**: `customer-feedback-analysis` (or preferred name)
   - **Runtime**: Select **Docker**
   - **Region**: Choose closest to your users (default: Oregon)
   - **Branch**: `main`
   - **Build Command**: (Leave empty - Docker handles it)
   - **Start Command**: (Leave empty - Docker handles it)
5. **Environment Variables** (optional):
   - Add any sensitive data needed (API keys, etc.)
6. **Plan**: Select **Free** tier (suitable for development)
7. Click **Create Web Service**

Render will automatically:
- Build your Docker image
- Deploy the application
- Provide a public URL

#### Option B: Using render.yaml Configuration File

1. Your `render.yaml` already exists in the repository
2. Go to https://dashboard.render.com
3. Click **"New +"** → **"Web Service"**
4. Connect your GitHub repository
5. Render will automatically detect and use `render.yaml` settings

### 4. Initial Build

- First deployment takes **5-10 minutes**
- Render builds the Docker image and deploys it
- You'll see build logs in real-time
- Once deployed, you'll get a public URL

### 5. Access Your Application

Your application will be available at:
```
https://customer-feedback-analysis.onrender.com
```

The exact URL will be provided in the Render dashboard.

## Post-Deployment

### Monitor Application

1. Check Render dashboard for logs and status
2. View real-time logs: **Dashboard** → **Your Service** → **Logs**
3. Monitor metrics: **Dashboard** → **Your Service** → **Metrics**

### Update Application

To deploy updates:

```bash
# Make code changes
git add .
git commit -m "Update description"
git push origin main
```

Render automatically redeploys when changes are pushed to the main branch.

### Manual Redeploy

From Render dashboard:
1. Go to your service
2. Click **"Manual Deploy"** → **"Deploy latest commit"**

### Restart Service

If your app freezes or needs a restart:
1. Go to your service in Render dashboard
2. Click **"Restart"** button

## Troubleshooting

### Application Won't Start

1. Check logs: **Dashboard** → **Logs** tab
2. Common issues:
   - **Import errors**: Check requirements.txt has all dependencies
   - **Memory issues**: Free tier has 512MB RAM - your app may need optimization
   - **Port binding**: Ensure Streamlit uses port 8501

### High Memory Usage

Streamlit apps can be memory-intensive. If on free tier:

1. Reduce data size in pipeline
2. Use `@st.cache_data` for large datasets
3. Consider upgrading to **Starter Plan** ($7/month)

### Session/State Issues

Streamlit requires specific configuration. The `.streamlit/config.toml` file handles this:

```toml
[server]
headless = true
port = 8501
```

### Dockerfile Build Fails

Ensure:
- Python 3.11 is available
- All dependencies in requirements.txt are compatible
- No local-only dependencies

### Large Data Files

If your `data/Amazon_Reviews.csv` is large:
- Render free tier has storage limits
- Consider:
  - Loading from cloud storage (S3, Google Cloud)
  - Using a smaller sample dataset
  - Upgrading to paid tier

## Configuration Options

### Change Python Version

Edit `runtime.txt`:
```
python-3.11
```

### Add Environment Variables

In Render dashboard:
1. Go to your service → **Environment**
2. Click **"Add Environment Variable"**
3. Set key-value pairs

Access in app.py:
```python
import os
api_key = os.getenv("API_KEY")
```

### Scale Up (Paid Tier)

For production:
1. Render dashboard → Your service → **Settings**
2. Upgrade plan from **Free** to **Starter** or **Standard**
3. Allocate more CPU and RAM
4. Enable auto-scaling

### Custom Domain

1. Render dashboard → Your service → **Settings**
2. Add custom domain
3. Update DNS records

## Cost Estimation

- **Free tier**: $0/month (512MB RAM, 0.5 CPU, auto-stops after 15 min inactivity)
- **Starter**: $7/month (1GB RAM, 1 vCPU)
- **Standard**: $25/month (2GB RAM, 2 vCPU)

## Next Steps

1. Test the deployed app thoroughly
2. Share the public URL with stakeholders
3. Monitor performance and user feedback
4. Plan upgrades if needed

## Support

- **Render Docs**: https://render.com/docs
- **Streamlit Deployment**: https://docs.streamlit.io/deploy
- **GitHub Issues**: Create issue in your repository
