import os
import json
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ═══════════════════════════════════════════════════════════════
# LOAD ENV & INIT CLIENT
# ═══════════════════════════════════════════════════════════════
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

if not GOOGLE_API_KEY:
    st.error("❌ GOOGLE_API_KEY tidak ditemukan. Silakan set di file .env atau environment variables.")
    st.info("Dapatkan API Key di: https://aistudio.google.com/app/apikey")
    st.stop()

client = genai.Client(api_key=GOOGLE_API_KEY)

# ═══════════════════════════════════════════════════════════════
# STREAMLIT CONFIG
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Gemini AAC Transcriber",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #4285f4; margin-bottom: 1rem; }
    .sub-header { font-size: 1.2rem; color: #666; margin-bottom: 2rem; }
    .stButton>button { width: 100%; height: 3rem; font-size: 1.1rem; background-color: #4285f4; color: white; }
    .transcript-box { background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 4px solid #4285f4; }
    .info-box { background-color: #e8f0fe; padding: 15px; border-radius: 8px; margin-bottom: 20px; }
    .success-box { background-color: #e6f4ea; padding: 15px; border-radius: 8px; border-left: 4px solid #34a853; }
    .warning-box { background-color: #fef3e8; padding: 15px; border-radius: 8px; border-left: 4px solid #f9ab00; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# GEMINI API FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def upload_audio_to_gemini(file_path: str, mime_type: str = None) -> types.File:
    """
    Upload audio file ke Google AI Studio / Gemini API.
    Gemini support AAC, MP3, WAV, AIFF, OGG, FLAC langsung tanpa konversi [^2^].
    """
    # Auto-detect mime type kalau tidak disediakan
    if mime_type is None:
        ext = Path(file_path).suffix.lower()
        mime_map = {
            '.aac': 'audio/aac',
            '.mp3': 'audio/mp3',
            '.wav': 'audio/wav',
            '.aiff': 'audio/aiff',
            '.ogg': 'audio/ogg',
            '.flac': 'audio/flac',
            '.m4a': 'audio/mp4',
        }
        mime_type = mime_map.get(ext, 'audio/aac')
    
    uploaded_file = client.files.upload(file=file_path, config={"mime_type": mime_type})
    return uploaded_file


def transcribe_audio_simple(uploaded_file: types.File, model: str, language_hint: str = None) -> str:
    """
    Transkripsi sederhana - hanya teks.
    """
    prompt = "Generate a complete and accurate transcript of the speech in this audio file."
    
    if language_hint:
        prompt += f" The audio is primarily in {language_hint} language."
    
    response = client.models.generate_content(
        model=model,
        contents=[
            types.Content(
                parts=[
                    types.Part(file_data=types.FileData(file_uri=uploaded_file.uri)),
                    types.Part(text=prompt)
                ]
            )
        ]
    )
    
    return response.text


def transcribe_with_structure(uploaded_file: types.File, model: str, include_timestamps: bool = False, 
                              include_speakers: bool = False, language: str = None) -> dict:
    """
    Transkripsi dengan structured output (JSON) - support timestamps dan speaker detection.
    """
    lang_instruction = f"The audio is in {language}." if language else "Auto-detect the language."
    
    prompt = f"""
    Transcribe the audio file with the following requirements:
    1. Provide accurate transcription of all speech content.
    2. {lang_instruction}
    3. Identify different speakers if multiple people are speaking.
    4. Provide timestamps for each segment in MM:SS format.
    
    Return as structured JSON with segments containing: text, speaker (if identifiable), timestamp.
    """
    
    if not include_timestamps:
        prompt = prompt.replace("Provide timestamps for each segment in MM:SS format.", 
                               "Do not include timestamps, just the text content.")
    
    if not include_speakers:
        prompt = prompt.replace("Identify different speakers if multiple people are speaking.", 
                               "Do not identify speakers, just transcribe the text.")
    
    # Define JSON schema untuk structured output
    schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "transcript": types.Schema(
                type=types.Type.STRING,
                description="The complete transcript text."
            ),
            "language": types.Schema(
                type=types.Type.STRING,
                description="Detected language of the audio."
            ),
            "segments": types.Schema(
                type=types.Type.ARRAY,
                description="List of transcribed segments.",
                items=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "text": types.Schema(type=types.Type.STRING),
                        "timestamp": types.Schema(type=types.Type.STRING),
                        "speaker": types.Schema(type=types.Type.STRING)
                    },
                    required=["text"]
                )
            )
        },
        required=["transcript", "language"]
    )
    
    response = client.models.generate_content(
        model=model,
        contents=[
            types.Content(
                parts=[
                    types.Part(file_data=types.FileData(file_uri=uploaded_file.uri)),
                    types.Part(text=prompt)
                ]
            )
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema
        )
    )
    
    # Parse JSON response
    try:
        return json.loads(response.text)
    except json.JSONDecodeError:
        return {"transcript": response.text, "language": "unknown", "segments": []}


def transcribe_advanced(uploaded_file: types.File, model: str, prompt_template: str = None,
                       language: str = None) -> str:
    """
    Transkripsi dengan custom prompt untuk kontrol penuh.
    """
    if prompt_template:
        prompt = prompt_template
    else:
        prompt = "Generate a transcript of the speech."
    
    if language:
        prompt += f" The audio is in {language} language."
    
    response = client.models.generate_content(
        model=model,
        contents=[prompt, uploaded_file]
    )
    
    return response.text


# ═══════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════
def main():
    st.markdown('<div class="main-header">🎙️ Gemini AAC Transcriber</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Transkripsi audio AAC/MP3/WAV ke teks dengan Google Gemini AI</div>',
                unsafe_allow_html=True)
    
    st.markdown(
        '<div class="info-box">'
        '✅ <b>Gemini API</b> mendukung AAC secara native tanpa konversi!<br>'
        '📁 Format: AAC, MP3, WAV, AIFF, OGG, FLAC, M4A<br>'
        '⏱️ Max duration: 9.5 hours per request [^2^]'
        '</div>',
        unsafe_allow_html=True
    )
    
    # ── Sidebar ──
    with st.sidebar:
        st.header("⚙️ Konfigurasi")
        
        model = st.selectbox(
            "Pilih Model Gemini",
            options=[
                "gemini-2.0-flash",
                "gemini-2.0-flash-lite",
                "gemini-2.5-flash-preview-05-20",
                "gemini-2.5-pro-preview-05-06"
            ],
            index=0,
            help="gemini-2.0-flash: cepat & hemat | gemini-2.5-pro: akurasi tertinggi"
        )
        
        st.divider()
        st.subheader("🎯 Mode Transkripsi")
        
        mode = st.radio(
            "Pilih mode:",
            options=["Simple (Text Only)", "Structured (JSON with timestamps)", "Advanced (Custom Prompt)"],
            index=0
        )
        
        language = st.text_input(
            "Kode Bahasa (opsional)",
            placeholder="Contoh: Indonesian, English, Japanese",
            help="Kosongkan untuk auto-detect"
        )
        
        include_timestamps = False
        include_speakers = False
        
        if mode == "Structured (JSON with timestamps)":
            include_timestamps = st.checkbox("Include Timestamps", value=True)
            include_speakers = st.checkbox("Include Speaker Labels", value=True)
        
        custom_prompt = None
        if mode == "Advanced (Custom Prompt)":
            custom_prompt = st.text_area(
                "Custom Prompt",
                placeholder="Contoh: 'Transcribe this meeting and extract all action items...'",
                help="Prompt kustom untuk kontrol penuh output"
            )
        
        st.divider()
        st.info(f"Model: {model}\nMode: {mode}")
    
    # ── Main ──
    uploaded = st.file_uploader(
        "📁 Upload File Audio",
        type=["aac", "mp3", "wav", "aiff", "ogg", "flac", "m4a"],
        help="Gemini support semua format ini secara native!"
    )
    
    if uploaded:
        # Info file
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Nama File", uploaded.name[:20] + "..." if len(uploaded.name) > 20 else uploaded.name)
        with col2:
            size_mb = len(uploaded.getvalue()) / (1024 * 1024)
            st.metric("Ukuran", f"{size_mb:.2f} MB")
        with col3:
            st.metric("Format", uploaded.type or "audio/aac")
        
        # Audio player
        st.audio(uploaded, format=uploaded.type or "audio/aac")
        
        st.divider()
        
        _, cbtn, _ = st.columns([1, 2, 1])
        with cbtn:
            transcribe_btn = st.button("🚀 Mulai Transkripsi", type="primary", use_container_width=True)
        
        if transcribe_btn:
            with st.spinner("Mengupload dan memproses audio..."):
                try:
                    # Simpan file temporarily
                    suffix = Path(uploaded.name).suffix
                    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                    tmp_file.write(uploaded.getvalue())
                    tmp_file.close()
                    
                    # Upload ke Gemini
                    with st.status("📤 Uploading ke Google AI Studio...", expanded=True) as status:
                        uploaded_file = upload_audio_to_gemini(tmp_file.name, uploaded.type)
                        status.update(label=f"✅ Upload selesai! File ID: {uploaded_file.name}", state="complete")
                    
                    # Transkripsi berdasarkan mode
                    with st.spinner("🧠 Gemini sedang mentranskripsi..."):
                        if mode == "Simple (Text Only)":
                            result = transcribe_audio_simple(
                                uploaded_file, 
                                model, 
                                language_hint=language if language else None
                            )
                            
                            st.success("✅ Transkripsi selesai!")
                            st.divider()
                            st.subheader("📝 Hasil Transkripsi")
                            st.markdown(f'<div class="transcript-box">{result}</div>', unsafe_allow_html=True)
                            st.code(result, language="text")
                            
                            # Download button
                            st.download_button(
                                "💾 Download TXT",
                                data=result,
                                file_name=f"{uploaded.name.rsplit('.', 1)[0]}_transcript.txt",
                                mime="text/plain"
                            )
                            
                        elif mode == "Structured (JSON with timestamps)":
                            result = transcribe_with_structure(
                                uploaded_file,
                                model,
                                include_timestamps=include_timestamps,
                                include_speakers=include_speakers,
                                language=language if language else None
                            )
                            
                            st.success("✅ Transkripsi selesai!")
                            st.divider()
                            
                            # Tampilkan info
                            st.subheader("📊 Informasi")
                            st.json({"language": result.get("language", "unknown"), "total_segments": len(result.get("segments", []))})
                            
                            # Tampilkan segments
                            st.subheader("📝 Hasil Transkripsi")
                            
                            if result.get("segments"):
                                for i, seg in enumerate(result["segments"]):
                                    with st.container():
                                        cols = st.columns([1, 4])
                                        with cols[0]:
                                            if include_timestamps and seg.get("timestamp"):
                                                st.caption(f"⏱️ {seg['timestamp']}")
                                            if include_speakers and seg.get("speaker"):
                                                st.markdown(f"**🎤 {seg['speaker']}**")
                                        with cols[1]:
                                            st.info(seg.get("text", ""))
                                        st.divider()
                            else:
                                st.markdown(f'<div class="transcript-box">{result.get("transcript", "")}</div>', 
                                          unsafe_allow_html=True)
                            
                            # Download JSON
                            st.download_button(
                                "💾 Download JSON",
                                data=json.dumps(result, indent=2, ensure_ascii=False),
                                file_name=f"{uploaded.name.rsplit('.', 1)[0]}_transcript.json",
                                mime="application/json"
                            )
                            
                        else:  # Advanced mode
                            result = transcribe_advanced(
                                uploaded_file,
                                model,
                                prompt_template=custom_prompt,
                                language=language if language else None
                            )
                            
                            st.success("✅ Transkripsi selesai!")
                            st.divider()
                            st.subheader("📝 Hasil Transkripsi")
                            st.markdown(f'<div class="transcript-box">{result}</div>', unsafe_allow_html=True)
                            st.code(result, language="text")
                            
                            st.download_button(
                                "💾 Download TXT",
                                data=result,
                                file_name=f"{uploaded.name.rsplit('.', 1)[0]}_transcript.txt",
                                mime="text/plain"
                            )
                    
                    # Cleanup temp file
                    os.unlink(tmp_file.name)
                    
                    # Info tentang file di Gemini
                    with st.expander("🔍 Detail File di Google AI Studio"):
                        st.write(f"File URI: {uploaded_file.uri}")
                        st.write(f"File Name: {uploaded_file.name}")
                        st.info("File akan disimpan sementara di Google AI Studio untuk pemrosesan.")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("💡 Tips: Pastikan GOOGLE_API_KEY valid dan file tidak corrupt.")
    
    else:
        st.info("👆 Upload file audio untuk memulai transkripsi")
        
        with st.expander("📖 Cara Penggunaan & Fitur"):
            st.markdown("""
            ### 🚀 Keunggulan Gemini API vs OpenAI:
            
            | Fitur | Gemini | OpenAI Whisper |
            |-------|--------|----------------|
            | **AAC Support** | ✅ Native, tanpa konversi | ❌ Perlu konversi ke MP3/WAV |
            | **Max Duration** | 9.5 hours [^2^] | 25 MB (~20-30 menit) |
            | **Speaker Detection** | ✅ Built-in | ❌ Limited |
            | **Timestamp** | ✅ MM:SS format | ✅ Word/segment level |
            | **Pricing** | Free tier tersedia | Pay-per-use |
            
            ### Langkah Penggunaan:
            1. **Dapatkan API Key** di [Google AI Studio](https://aistudio.google.com/app/apikey)
            2. **Set GOOGLE_API_KEY** di file `.env`
            3. **Upload file AAC** (atau MP3/WAV/FLAC/OGG)
            4. **Pilih mode transkripsi**:
               - *Simple*: Cepat, hanya teks
               - *Structured*: Dengan timestamp & speaker labels
               - *Advanced*: Custom prompt untuk kontrol penuh
            
            ### Format Audio yang Didukung [^2^]:
            - AAC (`audio/aac`) ⭐ **Native support!**
            - MP3 (`audio/mp3`)
            - WAV (`audio/wav`)
            - AIFF (`audio/aiff`)
            - OGG Vorbis (`audio/ogg`)
            - FLAC (`audio/flac`)
            """)
        
        with st.expander("🔑 Cara Mendapatkan API Key"):
            st.markdown("""
            1. Kunjungi [Google AI Studio](https://aistudio.google.com/app/apikey)
            2. Klik **"Create API Key"**
            3. Pilih project (atau buat baru)
            4. Copy API Key ke file `.env`:
               ```
               GOOGLE_API_KEY=your-api-key-here
               ```
            5. **Gratis!** Ada free tier dengan limit harian.
            """)


if __name__ == "__main__":
    main()
