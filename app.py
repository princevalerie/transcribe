import os
import base64
import tempfile
import subprocess
import glob
import shutil
from pathlib import Path

import streamlit as st
from openai import OpenAI, APIStatusError
from dotenv import load_dotenv
import imageio_ffmpeg

# ═══════════════════════════════════════════════════════════════
# FIX FFMPEG / FFPROBE PATH (auto-detect dari imageio-ffmpeg)
# ═══════════════════════════════════════════════════════════════
_FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
_FFMPEG_DIR = os.path.dirname(_FFMPEG_EXE)

# Cari ffprobe di folder yang sama dengan ffmpeg
_FFPROBE_CANDIDATES = glob.glob(os.path.join(_FFMPEG_DIR, "*ffprobe*"))
_FFPROBE_EXE = _FFPROBE_CANDIDATES[0] if _FFPROBE_CANDIDATES else shutil.which("ffprobe")

# Setup pydub untuk pakai binary dari imageio-ffmpeg
from pydub import AudioSegment
AudioSegment.converter = _FFMPEG_EXE
if _FFPROBE_EXE and os.path.exists(_FFPROBE_EXE):
    AudioSegment.ffprobe = _FFPROBE_EXE
else:
    AudioSegment.ffprobe = "ffprobe"

# ═══════════════════════════════════════════════════════════════
# LOAD ENV & INIT CLIENT
# ═══════════════════════════════════════════════════════════════
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ═══════════════════════════════════════════════════════════════
# STREAMLIT CONFIG
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AAC to Text Transcriber",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1f77b4; margin-bottom: 1rem; }
    .sub-header { font-size: 1.2rem; color: #666; margin-bottom: 2rem; }
    .stButton>button { width: 100%; height: 3rem; font-size: 1.1rem; }
    .transcript-box { background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 4px solid #1f77b4; }
    .info-box { background-color: #e8f4f8; padding: 15px; border-radius: 8px; margin-bottom: 20px; }
    .status-ok { color: #28a745; font-weight: bold; }
    .status-err { color: #dc3545; font-weight: bold; }
    .status-warn { color: #ffc107; font-weight: bold; }
    .fallback-banner { background-color: #fff3cd; border: 1px solid #ffc107; padding: 10px; border-radius: 5px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# AUDIO UTILITIES
# ═══════════════════════════════════════════════════════════════

def convert_aac_to_supported(input_path: str, output_format: str = "mp3") -> str:
    """
    Konversi AAC ke format yang didukung OpenAI (mp3/wav/m4a).
    Layer 1: pydub | Layer 2: subprocess ffmpeg (fallback)
    """
    out_file = tempfile.mktemp(suffix=f".{output_format}")

    # Layer 1: pydub
    try:
        audio = AudioSegment.from_file(input_path, format="aac")
        audio.export(out_file, format=output_format)
        return out_file
    except Exception:
        pass

    # Layer 2: subprocess ffmpeg langsung
    cmd = [
        _FFMPEG_EXE,
        "-y", "-hide_banner", "-loglevel", "error",
        "-i", input_path,
        "-vn", "-ar", "44100", "-ac", "2", "-b:a", "192k",
        out_file
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Gagal konversi audio:\n{result.stderr}")
    return out_file


def split_audio_ffmpeg(file_path: str, chunk_size_mb: int = 20) -> list:
    """
    Split file besar pakai ffmpeg segment. Tidak butuh ffprobe.
    """
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb <= chunk_size_mb:
        return [file_path]

    segment_minutes = max(int(chunk_size_mb / 1.4), 3)
    segment_time_sec = segment_minutes * 60

    temp_dir = tempfile.mkdtemp()
    base_name = Path(file_path).stem
    pattern = os.path.join(temp_dir, f"{base_name}_%03d.mp3")

    cmd = [
        _FFMPEG_EXE,
        "-y", "-hide_banner", "-loglevel", "error",
        "-i", file_path,
        "-f", "segment",
        "-segment_time", str(segment_time_sec),
        "-c", "copy",
        "-reset_timestamps", "1",
        pattern
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Gagal split audio:\n{result.stderr}")

    chunks = sorted([
        os.path.join(temp_dir, f)
        for f in os.listdir(temp_dir)
        if f.startswith(base_name) and f.endswith(".mp3")
    ])
    return chunks


def call_transcribe_api(file_path: str, model: str, response_format: str,
                        prompt: str = None, language: str = None, attempt_fallback: bool = True):
    """
    Kirim file ke OpenAI Audio Transcriptions API.
    Kalau 403 model_not_found, auto-fallback ke whisper-1.
    """
    with open(file_path, "rb") as audio:
        kwargs = {
            "model": model,
            "file": audio,
            "response_format": response_format,
        }
        # Prompt hanya untuk GPT-4o models
        if prompt and model in ("gpt-4o-transcribe", "gpt-4o-mini-transcribe"):
            kwargs["prompt"] = prompt
        if language:
            kwargs["language"] = language

        try:
            return client.audio.transcriptions.create(**kwargs), model
        except APIStatusError as e:
            if e.status_code == 403 and "model_not_found" in str(e).lower() and attempt_fallback:
                st.warning("⚠️ Model GPT-4o Transcribe tidak tersedia di project ini. Fallback ke `whisper-1`...")
                # Retry dengan whisper-1
                kwargs["model"] = "whisper-1"
                # Hapus prompt kalau whisper-1 (tidak support sama baik)
                if "prompt" in kwargs:
                    del kwargs["prompt"]
                return client.audio.transcriptions.create(**kwargs), "whisper-1"
            else:
                raise


def transcribe_diarize(file_path: str, response_format: str = "diarized_json"):
    """Transkripsi dengan speaker diarization."""
    with open(file_path, "rb") as audio:
        return client.audio.transcriptions.create(
            model="gpt-4o-transcribe-diarize",
            file=audio,
            response_format=response_format,
            extra_body={"chunking_strategy": "auto"}
        )


# ═══════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════
def main():
    st.markdown('<div class="main-header">🎙️ AAC to Text Transcriber</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Transkripsi AAC ke teks pakai OpenAI — Auto fallback ke whisper-1</div>',
                unsafe_allow_html=True)

    # ── Sidebar ──
    with st.sidebar:
        st.header("⚙️ Konfigurasi")

        # Status tools
        if os.path.exists(_FFMPEG_EXE):
            st.markdown('<span class="status-ok">✅ FFmpeg ready</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-err">❌ FFmpeg not found</span>', unsafe_allow_html=True)

        if _FFPROBE_EXE and os.path.exists(_FFPROBE_EXE):
            st.markdown('<span class="status-ok">✅ FFprobe ready</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-warn">⚠️ FFprobe fallback mode</span>', unsafe_allow_html=True)

        st.divider()

        # Model selection dengan info akses
        model = st.selectbox(
            "Pilih Model",
            options=[
                "gpt-4o-mini-transcribe",
                "gpt-4o-transcribe",
                "gpt-4o-transcribe-diarize",
                "whisper-1"
            ],
            index=3,  # Default whisper-1 (paling aman)
            help="whisper-1: tersedia semua | gpt-4o*: butuh akses khusus/beta"
        )

        if model != "whisper-1":
            st.markdown('<span class="status-warn">⚠️ Model ini butuh akses khusus. Akan auto-fallback ke whisper-1 kalau 403.</span>', unsafe_allow_html=True)

        # Format output
        if model == "gpt-4o-transcribe-diarize":
            fmt_opts = ["diarized_json", "json", "text"]
        elif model == "whisper-1":
            fmt_opts = ["json", "text", "srt", "verbose_json", "vtt"]
        else:
            fmt_opts = ["text", "json"]
        response_format = st.selectbox("Format Output", fmt_opts, index=0)

        language = st.text_input(
            "Kode Bahasa (ISO 639-1)",
            value=os.getenv("LANGUAGE", "id"),
            help="id=Indonesia, en=English, ja=Japanese, dst"
        )

        prompt = None
        if model in ("gpt-4o-transcribe", "gpt-4o-mini-transcribe"):
            p = st.text_area(
                "Prompt (opsional)",
                placeholder="Contoh: 'Percakapan tentang teknologi AI...'",
                help="Konteks untuk meningkatkan akurasi (hanya GPT-4o)"
            )
            prompt = p if p.strip() else None

    # ── Main ──
    st.markdown(
        '<div class="info-box">📁 <b>Upload file audio</b> (AAC/MP3/WAV/M4A/MP4, max 25MB per chunk).</div>',
        unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Pilih file",
        type=["aac", "m4a", "mp3", "wav", "mp4", "mpeg", "webm"],
        help="AAC akan dikonversi otomatis ke MP3"
    )

    if uploaded:
        c1, c2, c3 = st.columns(3)
        c1.metric("Nama", uploaded.name[:18] + "…" if len(uploaded.name) > 18 else uploaded.name)
        size_mb = len(uploaded.getvalue()) / (1024 * 1024)
        c2.metric("Ukuran", f"{size_mb:.2f} MB")
        c3.metric("Tipe", uploaded.type or "audio/aac")

        st.audio(uploaded, format="audio/aac")

        st.divider()
        _, cbtn, _ = st.columns([1, 2, 1])
        with cbtn:
            go = st.button("🚀 Mulai Transkripsi", type="primary", use_container_width=True)

        if go:
            with st.spinner("Memproses audio…"):
                try:
                    # Simpan upload ke temp file
                    tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".aac")
                    tmp_in.write(uploaded.getvalue())
                    tmp_in.close()

                    # Konversi ke MP3 (OpenAI nggak support AAC raw)
                    need_convert = uploaded.name.lower().endswith(".aac")
                    if need_convert:
                        st.info("🔄 Mengkonversi AAC → MP3…")
                        file_proc = convert_aac_to_supported(tmp_in.name, "mp3")
                    else:
                        file_proc = tmp_in.name

                    # Split kalau >25 MB
                    proc_size = os.path.getsize(file_proc) / (1024 * 1024)
                    if proc_size > 25:
                        st.warning(f"⚠️ File besar ({proc_size:.1f} MB), splitting…")
                        chunks = split_audio_ffmpeg(file_proc, chunk_size_mb=20)
                    else:
                        chunks = [file_proc]

                    # Proses tiap chunk
                    full_result = []
                    used_model = model
                    bar = st.progress(0, text="Memulai…")

                    for i, chunk in enumerate(chunks):
                        bar.progress((i) / len(chunks), text=f"Chunk {i+1}/{len(chunks)}…")

                        if model == "gpt-4o-transcribe-diarize":
                            # Diarize tidak support fallback otomatis di sini (karena format beda)
                            # Jika 403, user harus ganti manual ke whisper-1
                            res = transcribe_diarize(chunk, response_format)
                            used_model = "gpt-4o-transcribe-diarize"
                        else:
                            res, used_model = call_transcribe_api(
                                chunk, model, response_format,
                                prompt=prompt, language=language, attempt_fallback=True
                            )

                        if response_format == "text":
                            full_result.append(res)
                        else:
                            full_result.append(res.text if hasattr(res, "text") else str(res))

                        # Hapus chunk temp (kecuali original)
                        if chunk != file_proc:
                            os.unlink(chunk)

                    bar.progress(1.0, text="Selesai!")

                    # Banner info model yang benar-benar digunakan
                    if used_model != model:
                        st.markdown(f'<div class="fallback-banner">ℹ️ Digunakan model: <b>{used_model}</b> (fallback dari {model})</div>', unsafe_allow_html=True)
                    else:
                        st.success(f"✅ Transkripsi selesai! (Model: {used_model})")

                    st.divider()

                    # ── Tampilkan hasil ──
                    st.subheader("📝 Hasil Transkripsi")

                    if response_format == "diarized_json":
                        all_segments = []
                        for r in full_result:
                            if hasattr(r, 'segments'):
                                all_segments.extend(r.segments)
                            elif isinstance(r, dict) and 'segments' in r:
                                all_segments.extend(r['segments'])

                        if all_segments:
                            for seg in all_segments:
                                spk = getattr(seg, 'speaker', 'Unknown')
                                txt = getattr(seg, 'text', '')
                                s = getattr(seg, 'start', 0)
                                e = getattr(seg, 'end', 0)
                                col1, col2 = st.columns([1, 4])
                                col1.markdown(f"**🎤 {spk}**")
                                col1.caption(f"{s:.1f}s – {e:.1f}s")
                                col2.info(txt)
                        else:
                            st.json([r for r in full_result])

                    elif response_format in ("json", "verbose_json"):
                        st.json(full_result[0] if len(full_result) == 1 else full_result)

                    else:
                        final_text = " ".join(full_result) if isinstance(full_result, list) else full_result
                        if hasattr(final_text, 'text'):
                            final_text = final_text.text
                        st.markdown(f'<div class="transcript-box">{final_text}</div>', unsafe_allow_html=True)
                        st.code(final_text, language="text")
                        st.download_button(
                            "💾 Download TXT", data=str(final_text),
                            file_name=f"{uploaded.name.rsplit('.', 1)[0]}_transcript.txt",
                            mime="text/plain"
                        )

                    # Cleanup
                    os.unlink(tmp_in.name)
                    if need_convert and os.path.exists(file_proc):
                        os.unlink(file_proc)

                except APIStatusError as api_err:
                    if api_err.status_code == 403:
                        st.error(f"❌ Error 403: API Key tidak punya akses ke model ini.")
                        st.info("💡 Solusi: Pilih **whisper-1** di sidebar (tersedia untuk semua), atau cek billing/project di dashboard.openai.com")
                    else:
                        st.error(f"❌ API Error: {api_err}")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("💡 Pastikan API key valid di file `.env`.")

    else:
        st.info("👆 Upload file audio untuk mulai")
        with st.expander("📖 Cara Penggunaan & Troubleshooting"):
            st.markdown("""
            ### Langkah Penggunaan
            1. Upload file `.aac` (atau MP3/WAV/M4A)
            2. Pilih model — **whisper-1** paling aman (tersedia semua)
            3. Klik **Mulai Transkripsi**

            ### Error 403 / Model Not Found?
            Model `gpt-4o-transcribe`, `gpt-4o-mini-transcribe`, dan `gpt-4o-transcribe-diarize` **butuh akses khusus** dan belum tersedia untuk semua project OpenAI.

            **Solusi:**
            - Pilih **whisper-1** (default) — tersedia untuk semua API key
            - Kalau tetap mau GPT-4o, cek di [dashboard.openai.com](https://dashboard.openai.com) apakah model sudah di-enable

            ### Error FFmpeg?
            Sudah di-handle otomatis oleh `imageio-ffmpeg`. Tidak perlu install manual.
            """)


if __name__ == "__main__":
    main()
