import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font_path = 'C:/Windows/Fonts/MALGUNSL.ttf'  # 시스템에 설치된 한글 폰트 경로를 지정
font_prop = fm.FontProperties(fname=font_path)
plt.rc('font', family=font_prop.get_name())


# CSV 파일 읽기
file_path = '..\dataset\data_seoul.csv'  # CSV 파일 경로를 지정하세요
df = pd.read_csv(file_path)

# 데이터 인덱스를 시간 축으로 사용하여 그래프 그리기
plt.figure(figsize=(12, 6))

plt.plot(df.index, df['관측미세먼지'], label='관측미세먼지', color='blue', marker='o')
plt.plot(df.index, df['관측초미세먼지'], label='관측초미세먼지', color='red', marker='x')

plt.xlabel('시간 (인덱스)')
plt.ylabel('농도')
plt.title('관측미세먼지 및 관측초미세먼지 농도 변화')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
