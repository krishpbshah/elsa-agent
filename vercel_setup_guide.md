# Vercel Self-Hosting Setup Guide

This guide explains how to host your FastAPI application on Vercel yourself, using environment variables for security.

## 1. Files to Commit
Ensure your project directory contains these files. **Do NOT commit `elsaugc-6755d1abf892.json`**.

-   `fastapi_app.py` (Updated to read env vars)
-   `vercel.json` (Configuration)
-   `requirements.txt` (Dependencies)
-   `chat_interface.html` (Frontend)
-   `.gitignore` (Should include `.vercel` and `elsaugc-6755d1abf892.json`)

## 2. Setup Vercel Project
1.  Open your terminal in the project folder.
2.  Run `vercel login` (if not logged in).
3.  Run `vercel` to initialize the project.
    -   Set up and deploy? **Y**
    -   Which scope? (Select your account)
    -   Link to existing project? **N** (or Y if you want to overwrite)
    -   Project name? (e.g., `elsa-agent`)
    -   In which directory? **./**
    -   **Auto-detect settings?** It might detect Python. If asked for build command, leave empty or default.

## 3. Add Environment Variable (Crucial Step)
Your app needs the Google Cloud credentials to work. Since we aren't uploading the file, we put its **content** in an environment variable.

1.  Go to your Vercel Project Dashboard (the link provided after running `vercel`).
2.  Go to **Settings** > **Environment Variables**.
3.  Add a new variable:
    -   **Key**: `GOOGLE_APPLICATION_CREDENTIALS_JSON`
    -   **Value**: Paste the **entire content** of your `elsaugc-6755d1abf892.json` file here.
    -   Select environments: **Production**, **Preview**, **Development**.
4.  Click **Save**.

## 4. Redeploy
After adding the environment variable, you must redeploy for it to take effect.

1.  Run `vercel --prod` in your terminal.
2.  Wait for deployment to finish.
3.  Visit the URL provided.

## 5. Verification
-   Open the app URL.
-   Type "chibi elsa" in the chat.
-   If it generates an image, you are successfully hosted!
