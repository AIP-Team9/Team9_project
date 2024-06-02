import folium
import pandas as pd

# 미세먼지 농도 데이터
data = {
    'latitude': [37.5665, 35.1796, 36.3504],  # 예시: 서울, 부산, 대전
    'longitude': [126.9780, 129.0756, 127.3845],
    'pm2.5': [55, 120, 30]  # 미세먼지 농도 (예시)
}

# DataFrame으로 변환
df = pd.DataFrame(data)

# 지도 생성 (중심은 서울로 설정)
m = folium.Map(location=[37.5665, 126.9780], zoom_start=7)

# 색상 범위 설정 (미세먼지 등급에 따른 색상)
def get_color(pm):
    if pm <= 30:
        return 'blue'
    elif pm <= 80:
        return 'green'
    elif pm <= 150:
        return 'orange'
    else:
        return 'red'

# 데이터 포인트 추가
for idx, row in df.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=10,
        color=get_color(row['pm2.5']),
        fill=True,
        fill_color=get_color(row['pm2.5']),
        fill_opacity=0.7,
        popup=f"PM2.5: {row['pm2.5']}"
    ).add_to(m)



# 지도를 HTML 파일로 저장
m.save('pm_map.html')