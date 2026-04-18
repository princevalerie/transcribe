import os
import base64
import tempfile
from pathlib import Path

import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from pydub import AudioSegment

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Page configuration
st.set_page_config(
    page_title="AAC to Text Transcriber",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        height: 3rem;
        font-size: 1.1rem;
    }
    .transcript-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

def convert_aac_to_supported_format(input_path, output_format="mp3"):
    """
    Convert AAC file to a format supported by OpenAI API (mp3, mp4, mpeg, mpga, m4a, wav, webm)
    """
    audio = AudioSegment.from_file(input_path, format="aac")
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{output_format}")
    audio.export(temp_file.name, format=output_format)
    
    return temp_file.name

def transcribe_audio(file_path, model, response_format, prompt=None, language=None, chunking_strategy=None):
    """
    Transcribe audio using OpenAI API
    """
    with open(file_path, "rb") as audio_file:
        params = {
            "model": model,
            "file": audio_file,
            "response_format": response_format
        }
        
        # Add optional parameters
        if prompt and model in ["gpt-4o-transcribe", "gpt-4o-mini-transcribe"]:
            params["prompt"] = prompt
            
        if language:
            params["language"] = language
            
        if chunking_strategy and model == "gpt-4o-transcribe-diarize":
            params["extra_body"] = {"chunking_strategy": chunking_strategy}
        
        transcription = client.audio.transcriptions.create(**params)
        
    return transcription

def transcribe_with_diarization(file_path, model, response_format, known_speaker_names=None, known_speaker_references=None):
    """
    Transcribe audio with speaker diarization
    """
    def to_data_url(path):
        with open(path, "rb") as fh:
            return "data:audio/wav;base64," + base64.b64encode(fh.read()).decode("utf-8")
    
    with open(file_path, "rb") as audio_file:
        extra_body = {"chunking_strategy": "auto"}
        
        if known_speaker_names and known_speaker_references:
            extra_body["known_speaker_names"] = known_speaker_names
            extra_body["known_speaker_references"] = [to_data_url(ref) for ref in known_speaker_references]
        
        transcription = client.audio.transcriptions.create(
            model=model,
            file=audio_file,
            response_format=response_format,
            extra_body=extra_body
        )
    
    return transcription

def split_audio(file_path, chunk_size_mb=20):
    """
    Split large audio files into chunks (max 25MB per OpenAI limit)
    """
    audio = AudioSegment.from_file(file_path)
    
    # Calculate chunk duration in milliseconds
    # Approximate: 1MB ≈ 1 minute for MP3 at 128kbps
    chunk_duration_ms = (chunk_size_mb * 60 * 1000) // 1.5
    
    chunks = []
    for i in range(0, len(audio), chunk_duration_ms):
        chunk = audio[i:i + chunk_duration_ms]
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        chunk.export(temp_file.name, format="mp3")
        chunks.append(temp_file.name)
    
    return chunks

def main():
    # Header
    st.markdown('<div class="main-header">🎙️ AAC to Text Transcriber</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Transkripsi file audio AAC ke teks menggunakan AI GPT Models</div>', unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Konfigurasi")
        
        # Model selection
        model = st.selectbox(
            "Pilih Model",
            options=[
                "gpt-4o-mini-transcribe",
                "gpt-4o-transcribe", 
                "gpt-4o-transcribe-diarize",
                "whisper-1"
            ],
            index=0,
            help="gpt-4o-mini-transcribe: cepat & ekonomis | gpt-4o-transcribe: akurasi tinggi | gpt-4o-transcribe-diarize: dengan identifikasi speaker"
        )
        
        # Response format
        if model == "gpt-4o-transcribe-diarize":
            response_format = st.selectbox(
                "Format Output",
                options=["diarized_json", "json", "text"],
                index=0
            )
        elif model == "whisper-1":
            response_format = st.selectbox(
                "Format Output",
                options=["json", "text", "srt", "verbose_json", "vtt"],
                index=1
            )
        else:
            response_format = st.selectbox(
                "Format Output",
                options=["text", "json"],
                index=0
            )
        
        # Language
        language = st.text_input(
            "Kode Bahasa (ISO 639-1)",
            value=os.getenv("LANGUAGE", "id"),
            help="Contoh: id (Indonesia), en (English), ja (Japanese)"
        )
        
        # Prompt (for GPT-4o models)
        prompt = None
        if model in ["gpt-4o-transcribe", "gpt-4o-mini-transcribe"]:
            prompt = st.text_area(
                "Prompt (Opsional)",
                placeholder="Contoh: 'Transkripsi berikut adalah percakapan tentang teknologi AI...'",
                help="Bantu model dengan konteks untuk meningkatkan akurasi transkripsi"
            )
            if prompt == "":
                prompt = None
        
        # Diarization options
        enable_diarization = False
        if model == "gpt-4o-transcribe-diarize":
            enable_diarization = st.checkbox("Aktifkan Speaker Diarization", value=True)
        
        st.divider()
        st.info(f"**Model:** {model}\n**Format:** {response_format}\n**Bahasa:** {language}")
    
    # Main content
    st.markdown('<div class="info-box">📁 <b>Upload file AAC</b> (maksimal 25MB). File AAC akan dikonversi otomatis ke format yang didukung.</div>', unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Pilih file AAC",
        type=["aac", "m4a", "mp3", "wav", "mp4", "mpeg", "webm"],
        help="Format yang didukung: AAC, M4A, MP3, WAV, MP4, MPEG, WEBM"
    )
    
    if uploaded_file is not None:
        # Display file info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Nama File", uploaded_file.name[:20] + "..." if len(uploaded_file.name) > 20 else uploaded_file.name)
        with col2:
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
            st.metric("Ukuran", f"{file_size:.2f} MB")
        with col3:
            st.metric("Format", uploaded_file.type or "AAC")
        
        # Audio player
        st.audio(uploaded_file, format="audio/aac")
        
        # Transcribe button
        st.divider()
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            transcribe_button = st.button("🚀 Mulai Transkripsi", type="primary", use_container_width=True)
        
        if transcribe_button:
            with st.spinner("Memproses audio..."):
                try:
                    # Save uploaded file temporarily
                    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".aac")
                    temp_input.write(uploaded_file.getvalue())
                    temp_input.close()
                    
                    # Convert AAC to MP3 if needed
                    if uploaded_file.name.endswith('.aac'):
                        st.info("🔄 Mengkonversi AAC ke MP3...")
                        file_to_transcribe = convert_aac_to_supported_format(temp_input.name, "mp3")
                    else:
                        file_to_transcribe = temp_input.name
                    
                    # Check file size
                    file_size_mb = os.path.getsize(file_to_transcribe) / (1024 * 1024)
                    
                    if file_size_mb > 25:
                        st.warning(f"⚠️ File terlalu besar ({file_size_mb:.1f} MB). Membagi menjadi chunk...")
                        chunks = split_audio(file_to_transcribe)
                        
                        full_transcript = []
                        progress_bar = st.progress(0)
                        
                        for idx, chunk_path in enumerate(chunks):
                            st.text(f"Memproses chunk {idx + 1}/{len(chunks)}...")
                            
                            if model == "gpt-4o-transcribe-diarize" and enable_diarization:
                                result = transcribe_with_diarization(
                                    chunk_path, 
                                    model, 
                                    response_format
                                )
                            else:
                                result = transcribe_audio(
                                    chunk_path, 
                                    model, 
                                    response_format,
                                    prompt=prompt,
                                    language=language if language else None
                                )
                            
                            if response_format == "text":
                                full_transcript.append(result)
                            else:
                                full_transcript.append(result.text if hasattr(result, 'text') else str(result))
                            
                            progress_bar.progress((idx + 1) / len(chunks))
                            
                            # Clean up chunk
                            os.unlink(chunk_path)
                        
                        # Combine transcripts
                        if response_format == "diarized_json":
                            # For diarized JSON, we need to merge segments properly
                            combined_result = full_transcript
                        else:
                            combined_result = " ".join(full_transcript)
                    else:
                        # Single file transcription
                        if model == "gpt-4o-transcribe-diarize" and enable_diarization:
                            result = transcribe_with_diarization(
                                file_to_transcribe, 
                                model, 
                                response_format
                            )
                        else:
                            result = transcribe_audio(
                                file_to_transcribe, 
                                model, 
                                response_format,
                                prompt=prompt,
                                language=language if language else None
                            )
                        
                        combined_result = result
                    
                    # Clean up temp files
                    os.unlink(temp_input.name)
                    if uploaded_file.name.endswith('.aac'):
                        os.unlink(file_to_transcribe)
                    
                    # Display results
                    st.success("✅ Transkripsi selesai!")
                    st.divider()
                    
                    st.subheader("📝 Hasil Transkripsi")
                    
                    # Handle different response formats
                    if response_format == "diarized_json" and enable_diarization:
                        st.json(combined_result.segments if hasattr(combined_result, 'segments') else combined_result)
                        
                        # Display formatted diarized text
                        st.subheader("📋 Format Speaker")
                        if hasattr(combined_result, 'segments'):
                            for segment in combined_result.segments:
                                speaker = getattr(segment, 'speaker', 'Unknown')
                                text = getattr(segment, 'text', '')
                                start = getattr(segment, 'start', 0)
                                end = getattr(segment, 'end', 0)
                                
                                col_speaker, col_text = st.columns([1, 4])
                                with col_speaker:
                                    st.markdown(f"**🎤 {speaker}**")
                                    st.caption(f"{start:.1f}s - {end:.1f}s")
                                with col_text:
                                    st.info(text)
                    
                    elif response_format == "json":
                        st.json(combined_result)
                    
                    elif response_format == "verbose_json":
                        st.json(combined_result)
                    
                    else:
                        # Plain text
                        transcript_text = combined_result.text if hasattr(combined_result, 'text') else str(combined_result)
                        
                        st.markdown(f'<div class="transcript-box">{transcript_text}</div>', unsafe_allow_html=True)
                        
                        # Copy button
                        st.code(transcript_text, language="text")
                        
                        # Download button
                        st.download_button(
                            label="💾 Download Transkripsi",
                            data=transcript_text,
                            file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}_transcript.txt",
                            mime="text/plain"
                        )
                    
                    # Show raw response for debugging
                    with st.expander("🔍 Lihat Response Lengkap"):
                        st.write(combined_result)
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("💡 Tips: Pastikan API key valid dan file tidak corrupt.")
    
    else:
        # Show sample/info when no file uploaded
        st.info("👆 Upload file AAC untuk memulai transkripsi")
        
        with st.expander("📖 Cara Penggunaan"):
            st.markdown("""
            ### Langkah-langkah:
            1. **Upload File**: Pilih file AAC dari komputer Anda
            2. **Pilih Model**: 
               - `gpt-4o-mini-transcribe`: Cepat dan hemat biaya
               - `gpt-4o-transcribe`: Akurasi tinggi
               - `gpt-4o-transcribe-diarize`: Dengan identifikasi speaker
               - `whisper-1`: Model klasik Whisper
            3. **Konfigurasi**: Atur bahasa, format output, dan prompt (opsional)
            4. **Transkripsi**: Klik tombol "Mulai Transkripsi"
            
            ### Fitur:
            - ✅ Konversi otomatis AAC ke format yang didukung
            - ✅ Split file besar (>25MB) secara otomatis
            - ✅ Speaker diarization (identifikasi pembicara)
            - ✅ Multiple output formats (text, JSON, SRT, VTT)
            - ✅ Download hasil transkripsi
            
            ### Batasan:
            - Maksimal file size: 25MB per request (file besar akan di-split)
            - Format AAC akan dikonversi ke MP3
            """)

if __name__ == "__main__":
    main()
