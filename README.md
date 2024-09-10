# Story Videos Generator 🎥🖋️

**Story_Videos_Generator** is a Python project that automatically generates random suspenseful and horror stories, blending creativity with AI! Using **GPT-4o-mini** for storytelling and **LangChain** for orchestration, it creates chilling plots, while **Flux[Schnell]** generates eerie images to match the tone of the stories. 🖼️

The entire workflow, from story creation to video editing, is automated with **movie.py**, which produces high-quality videos ready for posting. Note that the inference may take some time since it runs via GitHub's endpoint (no GPU required!). 🎬

## Features
- **Story Generation**: Automatically creates captivating suspense and horror stories.
- **Image Generation**: Uses the powerful Flux[Schnell] model for generating vivid story-related images.
- **Video Editing**: Automatically composes the final video with minimal intervention, including audio and visual elements.
- **Ready for Sharing**: The final output is a polished video that you can instantly share or post.

## How to Use
1. Clone the repository and add your OpenAI API key and HuggingFace token to the `.env` file.
2. Run the `create_story.py` script.
3. A directory will be created with the title of your story. Inside, you'll find:
   - The generated script 📜
   - The images used 🖼️
   - The soundtrack 🎶
4. The final video will appear in the main directory, ready to watch or share! 🎥

## Collaboration
🚀 Want to improve the project or collaborate on similar ones? Feel free to reach out or explore my other [LLM projects](https://github.com/Azzedde). Together, we can make even better AI-generated content! 

