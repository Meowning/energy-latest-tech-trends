import time
from sources import motie, kaeri, kemri  # 각 기관별 크롤러 모듈

def run():
    print("Crawling started.")
    
    # 각 기관별 크롤러 실행
    motie.run()
    kaeri.run()
    kemri.run()
    
    print("Crawling completed.")

if __name__ == "__main__":
    run()
