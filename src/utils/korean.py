import platform
import matplotlib as mpl

def set_korean_font():
    system = platform.system()

    if system == 'Windows':
        mpl.rc('font', family='Malgun Gothic')
    elif system == 'Darwin':  # macOS
        mpl.rc('font', family='AppleGothic')
    else:
        mpl.rc('font', family='DejaVu Sans')  # 기본값 또는 리눅스용

    mpl.rc('axes', unicode_minus=False)
