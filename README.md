# Psychologist Assistant

A Streamlit-based application that helps psychologists with content creation, email drafting, and research summaries using LangGraph and OpenAI.

## Features

- Content Generation: Create professional content for LinkedIn, blogs, or social media
- Email Drafting: Generate professional emails for various purposes
- Research Summaries: Get concise summaries of psychological research topics

## Setup

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Running Locally

```bash
streamlit run psych_ui.py
```

## Deployment

This app is deployed on Streamlit Cloud. You can access it at: [Your Streamlit Cloud URL]

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)

## Contributing

Feel free to submit issues and enhancement requests! 