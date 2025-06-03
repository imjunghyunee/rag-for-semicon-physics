"""
Streamlit 앱 실행을 위한 래퍼 스크립트

사용법:
python run_streamlit.py
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Streamlit 앱 실행"""
    app_path = Path(__file__).parent / "streamlit_app.py"
    
    if not app_path.exists():
        print(f"Error: Streamlit app not found at {app_path}")
        sys.exit(1)
    
    print("🚀 Starting Semiconductor Physics RAG System...")
    print(f"📁 App location: {app_path}")
    print("🌐 Opening browser...")
    
    try:
        # Streamlit 실행
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(app_path),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Streamlit: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Shutting down Streamlit app...")

if __name__ == "__main__":
    main()
