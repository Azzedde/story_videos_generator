from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import FewShotPromptTemplate
import random
import requests  
import io
from PIL import Image
import os
import datetime
from tiktokvoice import tts
from moviepy.editor import ImageSequenceClip, concatenate_videoclips, AudioFileClip, CompositeVideoClip
import os
from pydub import AudioSegment
import freesound
from gradio_client import Client

# import from the .env file the OPEN_AI_KEY
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
hf_key = os.getenv("HF_token")


def generate_script(temperature, keyword):
    llm = OpenAI(model="gpt-4o-mini",api_key=api_key, temperature=temperature)
    initial_prompt = f"Write a suspenseful short story with fictional characters talking about {keyword} and give it a title. The story should be broken down into a 30-second timeline with time codes at every 5-second interval (except for the last part). Focus on building tension gradually, starting with a mysterious setting, followed by an unsettling event, and culminating in a cliffhanger. The tone should be eerie and the pacing should accelerate towards the end. Use vivid sensory details to enhance the suspense.\nHere's a 30-second short suspenseful story:"


    example_prompt = PromptTemplate.from_template("{prompt}\n{completion}")
    example = [
        {"prompt":initial_prompt,
        "completion":"""
    **"The Dark Stairway"**

    [0s-5s]
    Dr. Douglass finds himself standing in front of an old, creepy mansion with tall trees surrounding it. The wind is howling, and the creaking of the branches makes him feel uneasy. He takes a deep breath and push open the creaky door.

    [5s-10s]
    As he steps inside, the door slams shut behind him, and he hears the sound of locks clicking into place. He's plunged into darkness, except for a faint light flickering from upstairs. His heart starts racing as he realizes he's trapped.

    [10s-15s]
    The doctor slowly makes his way up the stairs, his eyes fixed on the light source. As he reaches the top, he sees a figure standing at the far end of the landing, illuminated by the dim glow of a single bulb. But as he takes another step forward...

    [15s-20s]
    ...the figure suddenly turns to face him. He freezes in terror, and his breath catches in his throat. It's a woman with a twisted grin on her face, her eyes black as coal. She takes a slow step closer, her voice barely above a whisper: "Welcome home..."

    [20s-30s]
    His heart is pounding like a drum, and he tries to take a step back, but his feet feel rooted to the spot. The woman's grin grows wider, and she raises her hand, as if reaching for something...
    """}
    ]



    prompt = FewShotPromptTemplate(
        examples=example,
        example_prompt=example_prompt,
        suffix="{input}",
        input_variables=["input"],
    )
    """Generates a script using the model."""

    message = llm.invoke(prompt.invoke({"input": initial_prompt}).to_string())
    return message

def parse_script(script):
    """Parses the script and extracts the title, timeline, and content."""
    title = script.split("**")[1]
    # we extract the text of each event in the timeline without the time codes
    timeline = [event.split("\n")[1] for event in script.split("[")[1:]]

    
    return title, timeline

def generate_image_generation_prompt(timeline_event):
    llm = OpenAI(model="gpt-4o-mini",api_key=api_key)
    """Generates an image generation prompt based on the timeline event."""
    initial_prompt = f"Generate one sentence that summarizes the following event, the sentence will be used as a caption to describe an image related to the event, avoid talking about texts and discussions, if the event is a discussion try to describe the persons talking as best as you can:\n{timeline_event}\nHere is the description sentence:"
    example_prompt = PromptTemplate.from_template("Generate one sentence that summarizes the following event, the sentence will be used as a caption to describe an image related to the event:\n{example_event}\nHere is the description sentence: {completion}")
    example = [
        {"example_event":"You find yourself standing in front of an old, creepy mansion with tall trees surrounding it. The wind is howling, and the creaking of the branches makes you feel uneasy. You take a deep breath and push open the creaky door.",
        "completion":"A man standing in front of a creepy mansion with tall trees surrounding it and a creaky door."},
        {"example_event":"As you step inside, the door slams shut behind you, and you hear the sound of locks clicking into place. You're plunged into darkness, except for a faint light flickering from upstairs. Your heart starts racing as you realize you're trapped.",
         "completion": "A man entering a dark room with a faint light flickering from upstairs."},
         {"example_event":"You slowly make your way up the stairs, your eyes fixed on the light source. As you reach the top, you see a figure standing at the far end of the landing, illuminated by the dim glow of a single bulb. But as you take another step forward...", 
          "completion":"A very dark room with a dark figure standing at the end of the landing."},
          {"example_event":"John told Mary, 'We need to hurry up in order to not be late for the meeting.' Mary replied, 'I know, but I can't find my keys anywhere.' John said, 'Don't worry, we'll find them together.'",
           "completion":"a man discussing with a worried woman about finding keys."},
    ]
    prompt = FewShotPromptTemplate(
        examples=example,
        example_prompt=example_prompt,
        suffix="{input}",
        input_variables=["input"],
    )

    message = llm.invoke(prompt.invoke({"input": initial_prompt}).to_string())
    
    return message

def generate_image(image_prompt, title):
    """Generates an image based on the image prompt."""
    API_URL = "https://api-inference.huggingface.co/models/XLabs-AI/flux-RealismLora"
    headers = {"Authorization": f"Bearer {hf_key}"}
    payload = {
        "inputs": image_prompt,
    }


    response = requests.post(API_URL, headers=headers, json=payload)
    image_bytes = response.content
    # You can access the image with PIL.Image for example

    image = Image.open(io.BytesIO(image_bytes))
    # create a directory in the current directory named as the variable title
    
    
    now = datetime.datetime.now()
    # save the image in the directory
    image.save(f"./{title}/{now}.jpg")
    return None



# def get_soundeffect(keyword, title):
#     """Downloads a sound effect from Freesound based on a keyword."""
#     client = freesound.FreesoundClient()
#     client.set_token("0kbUjvuEPiwhUsPjSEigezFZjweORZ2CoZaBxoGo","token")
#     results = client.text_search(query=keyword,fields="id,name,previews")
#     save_path = f'./{title}/{results[0].name}.mp3'
#     print(save_path)
#     print(results[0].name)
#     results[0].retrieve_preview(f"./{title}/",results[0].name+".mp3")
#     return save_path

# def generate_sound_effect_prompt(event):
#     llm = OllamaLLM(model="llama3.1", device="cuda", temperature=0.1)
#     """Generates a sound effect generation prompt based on the timeline event."""
#     initial_prompt = f"Generate a sentence that describes the most suitable sound effect of following event:\n{event}\nHere is the sound effect:"
#     example_prompt = PromptTemplate.from_template("Generate a sentence that describes the most suitable sound effect of following event:\n{example_event}\nHere is the sound effect: {completion}")
#     example = [
#         {"example_event":"Professor Emma Taylor stood at the edge of the dense jungle, her eyes scanning the underbrush for any sign of movement. The air was thick with the sounds of insects and animals, but one creature in particular had been rumored to inhabit this terrain: the legendary Kala Serpent.",
#         "completion":"wild jungle sounds"},
#         {"example_event":"...and Emma's heart skipped a beat as she caught sight of a massive serpent slithering through the underbrush. Its scales glistened in the dappled light, and its eyes seemed to gleam with an otherworldly intelligence.",
#          "completion": "snake hissing"},
#          {"example_event":"Renowned journalist, Emma Taylor, sat in her dimly lit office, surrounded by stacks of dusty papers and old typewriters. She stared at an antique pen lying on her desk, its intricate design glinting in the faint light. The air was thick with the scent of old books and dust.", 
#           "completion":"library ambience"},
#         {"example_event":"Emily opened the book, releasing a faint scent of old parchment and forgotten knowledge. As she began to read, the words seemed to shift on the page, like living, breathing creatures.",
#          "completion":"reading through a book"}
#     ]
#     prompt = FewShotPromptTemplate(
#         examples=example,
#         example_prompt=example_prompt,
#         suffix="{input}",
#         input_variables=["input"],
#     )

#     message = llm.invoke(prompt.invoke({"input": initial_prompt}).to_string())
    
#     return message

# def generate_sound_effect(sound_effect_prompt, title, duration):
#     """Generates a sound effect based on the sound effect prompt."""
#     client = Client("https://fffiloni-audiogen.hf.space/--replicas/vd9sp/")
#     result = client.predict(
#             sound_effect_prompt,	# str  in 'audio prompt' Textbox component
#             6,	# float (numeric value between 5 and 10) in 'Duration' Slider component
#             api_name="/infer"
#     )
#     result.save(f"./{title}/sound_effect.mp3")

voices = ['en_us_001',                  # English US - Female (Int. 1)
    'en_us_002',                  # English US - Female (Int. 2)
    'en_us_006',                  # English US - Male 1
    'en_us_007',                  # English US - Male 2
    'en_us_009',                  # English US - Male 3
    'en_us_010', ]

voice = random.choice(voices)

trendy_keywords = ["a mysterious witch that poisons children","a weird politician that kidnaps women","a dangerous monster that eats humans","a haunted house that traps people","a cursed doll that kills its owners","a creepy clown that terrorizes a town","a ghost ship that appears in the night","a cursed forest that drives people insane","a haunted hotel that traps its guests", "a possessed door that leads to another dimension","a weird adventure to another realm through a trap","a haunted classroom that makes everyone crazy","a woman that goes to dates and kidnaps the men she meet", ""]
keyword = random.choice(trendy_keywords)
temperature = random.uniform(0.5, 1.0)
print(f"Generating a suspenseful short story about {keyword} with a temperature of {temperature}...")
script = generate_script(temperature, keyword)
print("Script generated successfully!")
title, timeline = parse_script(script)
title = title.replace('"',"")
os.makedirs(title, exist_ok=True)
# save the title and the timeline in a text file
with open(f"{title}/story.txt", "w") as f:
    f.write(title + "\n\n")
    for event in timeline:
        f.write(event + "\n")
now = datetime.datetime.now()
i=0
for event in timeline:
    image_prompt = generate_image_generation_prompt(event)
    generate_image(image_prompt,title)
    tts(event.replace("..."," "),voice,f"{title}/{i}.mp3",False)
    print("Image and voice generated successfully!")
    i+=1

print("All images and voices generated successfully!")



# Define the directory containing the images
image_directory = f'./{title}'

# Get the list of images in the directory
image_files = sorted([os.path.join(image_directory, img) for img in os.listdir(image_directory) if img.endswith(('.png', '.jpg', '.jpeg'))])

# Get the list of voice files in the directory
voice_files = sorted([os.path.join(image_directory, img) for img in os.listdir(image_directory) if img.endswith('.mp3')])


# Define a list of durations for each voice file
durations = []

for voice_file in voice_files:
    voice = AudioSegment.from_file(voice_file)
    duration = len(voice) / 1000  # Convert from milliseconds to seconds
    durations.append(duration)

# Create a list of video clips, each corresponding to an image with its respective duration and audio
clips = []
for image_file, voice_file, duration in zip(image_files, voice_files, durations):
    # Create a video clip from the image
    clip = ImageSequenceClip([image_file], durations=[duration])
    
    # Add the corresponding audio file
    audio = AudioFileClip(voice_file)
    clip = clip.set_audio(audio)
    
    clips.append(clip)

# Concatenate all the clips into a single video
final_clip = concatenate_videoclips(clips, method="compose")


# Write the video file
output_file = f'{title}.mp4'
final_clip.write_videofile(output_file, codec='libx264', fps=24)
