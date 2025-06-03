"""
Streamlit 실행 전 의존성 문제를 자동으로 해결하는 스크립트
"""
import subprocess
import sys
import importlib.util

def check_and_install_package(package_name, pip_name=None):
    """패키지가 설치되어 있는지 확인하고 없으면 설치"""
    if pip_name is None:
        pip_name = package_name
    
    try:
        importlib.import_module(package_name)
        print(f"✅ {package_name} is already installed")
        return True
    except ImportError:
        print(f"❌ {package_name} not found, installing {pip_name}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
            print(f"✅ {pip_name} installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {pip_name}: {e}")
            return False

def fix_langchain_dependencies():
    """LangChain 의존성 문제 해결"""
    print("🔧 Checking LangChain dependencies...")
    
    # 새로운 langchain-huggingface 패키지 확인
    if not check_and_install_package("langchain_huggingface", "langchain-huggingface"):
        print("⚠️ langchain-huggingface installation failed, checking fallback...")
        # fallback으로 langchain-community 확인
        if not check_and_install_package("langchain_community", "langchain-community"):
            print("❌ Both langchain-huggingface and langchain-community failed to install")
            return False
    
    return True

def fix_streamlit_dependencies():
    """Streamlit 관련 의존성 확인"""
    print("🔧 Checking Streamlit dependencies...")
    
    packages = [
        ("streamlit", "streamlit>=1.28.0"),
        ("nest_asyncio", "nest-asyncio>=1.5.0"),
    ]
    
    for package_name, pip_name in packages:
        if not check_and_install_package(package_name, pip_name):
            return False
    
    return True

def main():
    """메인 의존성 수정 함수"""
    print("🚀 Starting dependency fix process...")
    
    success = True
    
    # Streamlit 의존성 확인
    if not fix_streamlit_dependencies():
        success = False
    
    # LangChain 의존성 확인
    if not fix_langchain_dependencies():
        success = False
    
    if success:
        print("✅ All dependencies are ready!")
        print("🎉 You can now run: streamlit run streamlit_app.py")
    else:
        print("❌ Some dependencies failed to install")
        print("🔧 Please manually install missing packages")
    
    return success

if __name__ == "__main__":
    main()
