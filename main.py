from fastapi import FastAPI, Request
# from fastapi.responses import JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# from starlette.applications import Starlette
# from starlette.middleware import Middleware
# from starlette.routing import Mount
from engineio.payload import Payload
from pydub import AudioSegment
from pydub.utils import mediainfo
from contextlib import asynccontextmanager
from query_data import get_response
import grpc
import audio2face_pb2
import audio2face_pb2_grpc
import numpy as np
import time
import socketio
import datetime
import riva.client
import riva.client.audio_io
import requests
import aiohttp
import asyncio
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
Payload.max_decode_packets = 500
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins='*')
sio_app = socketio.ASGIApp(socketio_server=sio)

# ELEVEN LABS PARAMETERS
headers= {
  "Accept": "audio/wav",
  "Content-Type": "application/json",
  "xi-api-key": "cdadb7bd2efb978c726a897f96cadd1a"
}

# A2F COMPONENTS
instance = "/World/audio2face/CoreFullface"
StreamLiveLink =  "/World/audio2face/StreamLivelink"
BlendShapeSolver =  "/World/audio2face/BlendshapeSolve"
a2f_player_streaming = "/World/audio2face/audio_player_streaming"
a2f_player_regular = "/World/audio2face/Player"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # A2F API VARIABLES
    # file_path = r"D:\Unreal_Server\mark_regular.usd"
    # a2f_current_audio_files_folder = "D:/Unreal_Server/"
    # url_load = 'http://172.16.15.209:8011/A2F/USD/Load'
    # url_activatestreamlivelink = 'http://172.16.15.209:8011/A2F/Exporter/ActivateStreamLivelink'
    # url_a2e_streaming = 'http://172.16.15.209:8011/A2F/A2E/EnableStreaming'
    # url_a2e_autogen_onchange = 'http://172.16.15.209:8011/A2F/A2E/EnableAutoGenerateOnTrackChange'
    # url_set_track_loop = 'http://172.16.15.209:8011/A2F/Player/SetLooping'
    # url_get_current_track = 'http://172.16.15.209:8011/A2F/Player/GetCurrentTrack'
    # url_set_current_track = 'http://172.16.15.209:8011/A2F/Player/SetTrack'
    # url_get_root_path = 'http://172.16.15.209:8011/A2F/Player/GetRootPath'
    # url_set_root_path = 'http://172.16.15.209:8011/A2F/Player/SetRootPath'
    # url_setstreamlivelinksettings = 'http://172.16.15.209:8011/A2F/Exporter/SetStreamLivelinkSettings'
    # url_get_tracks = 'http://172.16.15.209:8011/A2F/Player/GetTracks'
    
    # # LOADING A2F MODEL
    # body_load_usd = {
    # 'file_name': file_path
    # }
    # response_load = requests.post(url=url_load , json=body_load_usd)
    # print(f"BOOT LOAD: {response_load.json()}")
    
    # #set root path for audio files
    # body_root_path = {
    #     "a2f_player": a2f_player_regular,
    #     "dir_path" : a2f_current_audio_files_folder
    # }
    # response_root_path = requests.post(url=url_set_root_path , json=body_root_path)
    # print(f"ROOT PATH SET: {response_root_path.json()}")
    
    # #get tracks in root path
    # body_get_tracks = {
    #     "a2f_player": a2f_player_regular
    # }
    # response_tracks = requests.post(url=url_get_tracks , json=body_get_tracks)
    # print(f"TRACKS LIST: {response_tracks.json()}")
    
    # #set track
    # body_set_track = {
    #     "a2f_player": a2f_player_regular,
    #     "file_name": 'output.wav',
    #     "time_range": [
    #          0,
    #         -1
    #     ]
    # }
    
    # response_set_track = requests.post(url=url_set_current_track , json=body_set_track)
    # print(f"BODY SET TRACK: {response_set_track.json()}")
    
    # #set track loop to false
    # body_set_track_loop = {
    #     "a2f_player": a2f_player_regular,
    #     "loop_audio": False
    # }
    # response_tracks_loop = requests.post(url=url_set_track_loop , json=body_set_track_loop)
    # print(f"TRACKS LOOP {response_tracks_loop.json()}")
    
    
    # #set live link settings
    # body_live_ink_settings = {
    # "node_path": StreamLiveLink,
    # "values": {"enable_audio_stream": True ,  "livelink_host": '172.16.15.207'  , "enable_gaze": False , "enable_idle_head": False }
    # }
    # response_live_link_settings = requests.post(url=url_setstreamlivelinksettings , json=body_live_ink_settings)
    # print(f"LIVELINK SETTING {response_live_link_settings.json()}")
    
    # #enablae A2E auto gen on track change
    # body_a2e_auto_gen = {
    # "a2f_instance": instance ,
    # "enable": True 
    # }
    # response_a2e_auto_gen = requests.post(url=url_a2e_autogen_onchange , json=body_a2e_auto_gen)
    # print(F"EANBLE A2E AURO GEN ON CHANGE {response_a2e_auto_gen.json()}")
    
    # #enable A2E streaming
    # body_a2e_stream = {
    # "a2f_instance": instance ,
    # "enable": True 
    # }
    # response = requests.post(url=url_a2e_streaming , json=body_a2e_stream)
    # print(F"ENABLE A2E STREAMING {response.json()}")
    
    
    # #activate live link
    # body_activate_live_link = {
    # "node_path": StreamLiveLink ,
    # "value": True
    # }
    # response_activate_live_link = requests.post(url=url_activatestreamlivelink , json=body_activate_live_link)
    # print(f"ACTIVATE STREAMLINK {response_activate_live_link.json()}")
    
    yield


app = FastAPI(lifespan=lifespan)

# RIVA PARAMETERS
boosted_words = ["swift", "vision", "aqua", "gaze", "glasses", "mind", "lens", "terra"]
boosted_lm_score = 20.0
riva_uri = '172.16.15.217:50051'  # ip address of NVIDIA RIVA ASR and TTS
offline_output_file = 'test_audio.wav'
language_code = 'en-US'
sample_rate = 48000
nchannels = 1
sampwidth = 2

product_list = None
product_list_lock = asyncio.Lock()
player_position = None
player_position_lock = asyncio.Lock()


def push_audio_track(url, audio_data, samplerate, instance_name):
    block_until_playback_is_finished = True  # ADJUST
    with grpc.insecure_channel(url) as channel:
        stub = audio2face_pb2_grpc.Audio2FaceStub(channel)
        request = audio2face_pb2.PushAudioRequest()
        request.audio_data = audio_data.astype(np.float32).tobytes()
        request.samplerate = samplerate
        request.instance_name = instance_name
        request.block_until_playback_is_finished = block_until_playback_is_finished
        print("Sending audio data...")
        response = stub.PushAudio(request)
        if response.success:
            print("SUCCESS")
        else:
            print(f"ERROR: {response.message}")
    print("Closed channel")


def convertToAudioAndPlay(question, language):
    # Sleep time emulates long latency of the request
    sleep_time = 0.2  # ADJUST

    # URL of the Audio2Face Streaming Audio Player server (where A2F App is running)
    a2f_url = '172.16.15.209:50051'  # ADJUST to where the Audio2Face instance is running 

    # Prim path of the Audio2Face Streaming Audio Player on the stage (were to push the audio data)
    instance_name = '/World/audio2face/audio_player_streaming'
    
    auth = riva.client.Auth(uri=riva_uri)
    tts_service = riva.client.SpeechSynthesisService(auth)

    language_code = 'en-US'
    sample_rate = 44100
    print("inside 1")
    resp = tts_service.synthesize(question, language_code=language_code , sample_rate_hz=sample_rate , encoding=riva.client.AudioEncoding.LINEAR_PCM , voice_name="English-US.Female-1" )
    print("inside 2")
    audio_data = np.frombuffer(resp.audio, dtype=np.int16)
    dtype = np.dtype('float32')
    i = np.iinfo(audio_data.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    audio_data = (audio_data.astype(dtype) - offset) / abs_max

    print(f"Sleeping for {sleep_time} seconds")
    time.sleep(sleep_time)
    push_audio_track(a2f_url, audio_data, sample_rate, instance_name)
    return f"Audio pushed to A2F"


# API CALL FOR A2F
def a2f_api_call():
    url_play_track = 'http://172.16.15.209:8011/A2F/Player/Play'
    url_set_current_track = 'http://172.16.15.209:8011/A2F/Player/SetTrack'
    
    body_set_track = {
        "a2f_player": a2f_player_regular,
        "file_name": 'output.wav',
        "time_range": [
             0,
            -1
        ]
    }
    
    response_set_track = requests.post(url=url_set_current_track , json=body_set_track)
    print(f"A2F FUNCTION RESPONSE: {response_set_track.json()}")
    body = {
        "a2f_player": a2f_player_regular
    }
    response = requests.post(url=url_play_track , json=body)
    print(f"PLAY TRACK: {response.json()}")
  

# TRANSCRIPTION TO RASA FOR DIALOGUE
async def send_to_rasa(sid, response):
    res = {
        "sender": sid,
        "message": response
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url="http://172.16.15.217:5005/webhooks/rest/webhook", json=res) as resp:
            rasa_output = await resp.json()
    
    return rasa_output


def inspect_audio_file(file_path):
    info = mediainfo(file_path)
    print("AUIDO METADATA")
    for key, val in info.items():
        print(f"{key}: {val}")


# TRANSCRIPTION TO ELEVENLABS FOR AUDIO
def eleven_labs_api(text):
    str(text)
    
    voice_id= "XrExE9yKIg1WjnnlVkGX"
    CHUNK_SIZE = 1024
    model_id = "eleven_turbo_v2"
    url_eleven_labs = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    try:
        payload = {
        "model_id": model_id,
        "text": text,
        "voice_settings": {
            "similarity_boost": 0.5,
            "stability": 0.75,
            "use_speaker_boost": True
            }
        }
        response = requests.post(url=url_eleven_labs, json=payload, headers=headers)
    except Exception as e:
        print(f"ERROR IN ELEVEN LABS API: {e}") 
        return
    
    audio_bytes = b"" 
    with open('output.mp3', 'wb') as f:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                audio_bytes += chunk
                
    # with open('output.wav', 'wb') as f:
    #     for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
    #         if chunk:
    #             f.write(chunk)

    # mp3_file = AudioSegment.from_file("output.mp3", format="mp3")
    # wav_file = mp3_file.set_frame_rate(44100).set_channels(1)
    # wav_file.export("output.wav", format="wav")
    
    return str(audio_bytes)


@app.get("/product_list")
async def get_product_list():
    global product_list
    # async with product_list_lock:
    return {"product_list": product_list}


@app.post("/rag")
async def rag(request: Request):
    nlu = await request.json()
    rag_response = get_response(nlu)
    
    return  rag_response


@app.get("/player_position")
async def get_player_position():
    global player_position
    # async with product_list_lock:
    return {"player_position": player_position}


# EVENT HANDLER FOR CONNECTION
@sio.event
async def connect(sid, auth):
    print(f'{sid}: connected on {datetime.datetime.now().time()}\n')


# EVENT HANDLER FOR GETTING PRODUCT DATA FROM UNREAL
@sio.event
async def get_products(sid, data):
    global product_list
    try:
        async with product_list_lock:
            product_list = data
        print(f'PRODUCT LIST FROM SOCKETIO : {product_list}\n')
        await sio.emit('server_response', {"data" : "products received"}, to=sid)
        
        return product_list
    except Exception as error:
        print(f"Error in getting data : {error}")
        

# EVENT HANDLER FOR GETTING POSITION
@sio.event
async def get_position(sid, data):
    global player_position
    try:
        async with player_position_lock:
            player_position = data
        print(f'POSITION FROM SOCKETIO : {player_position}\n')
        await sio.emit('server_response', {"data" : "position received"}, to=sid)
        
        return player_position
    except Exception as error:
        print(f"Error in getting data : {error}")


# EVENT HANDLER FOR COMMUNICATION WITH UNREAL
@sio.event
async def send_message(sid, data):
    global audio_bytes
    try:        
        # RIVA CONFIGURATION
        auth = riva.client.Auth(uri=riva_uri)
        asr_service = riva.client.ASRService(auth)
        offline_config = riva.client.RecognitionConfig(
            encoding=riva.client.AudioEncoding.LINEAR_PCM,
            max_alternatives=1,
            enable_automatic_punctuation=False,
            verbatim_transcripts=False,
            profanity_filter=False,
            sample_rate_hertz=16000,
            audio_channel_count=1,
            language_code='en-US'
        )
        riva.client.add_word_boosting_to_config(
        offline_config, boosted_words, boosted_lm_score)
        try:
            response = asr_service.offline_recognize(data, offline_config)
            # print(f"RESPONSE : {response}")
            if (len(response.results[0].alternatives) <= 0):
                print("No data found")
            else:
                final_response = ""
                for resp in response.results:
                    final_response = final_response + resp.alternatives[0].transcript
                print(f"\nTRANSCRIPTION : {final_response}\n")

                # SENDING/RECEIVING TRANSCRIPTION TO RASA
                rasa_response = await send_to_rasa(sid, final_response)
                print(f"RASA RESPONSE : {rasa_response}")

                rasa_text = rasa_response[0]['text']
                rasa_json = rasa_response[1]['custom']

                eleven_labs_api(rasa_text)
                # audio_bytes = eleven_labs_api(rasa_text)
                # a2f_api_call()
                # print(convertToAudioAndPlay(rasa_text, 'en-US'))
                
                print(f"RASA RESPONSE TO ELEVEN LABS : {rasa_text}")
                print(f"RASA RESPONSE TO UNREAL : {rasa_json}")
                # print(f"AUDIO BYTES FROM ELEVEN LABS : {audio_bytes}")
                # SENDING THE REPONSE TO UNREAL
                await sio.emit('server_response', rasa_json, to=sid)
                # await sio.emit('audio_bytes', audio_bytes, to=sid)
                print('emitted')
        except Exception as e:
            print(f"Exception Occured Internal : {e}")
    except Exception as error:
        print(f"Exception Occured External : {error}")

# EVENT HANDLER FOR DISCONNECTION
@sio.event
async def disconnect(sid):
    print(f'{sid} Disconnected on {datetime.datetime.now().time()}')


# middleware = [
#     Middleware(
#         CORSMiddleware,
#         allow_origins=['*'],
#         allow_methods=['*'],
#         allow_headers=['*'],
#         allow_credentials=True
#     )
# ]

app.mount('/', app=sio_app)
app.mount('/api', app=app)


# RUN THE APPLICATION, USE THIS COMMAND
# uvicorn main:app --host 172.16.15.217 --reload