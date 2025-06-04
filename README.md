# rag-for-semicon-physics

1. pip install -r requirements.txt 실행

    -   만약 streamlit에서 의존성 문제가 발생한다면 requirements_streamlit.txt로도 다운
    -   그래도 streamlit이 안된다면 python fix_dependencies.py 실행
    -   만약 위 두 단계 없이 바로 streamlit과 python main.py가 실행된다면 위 두 파일 requirements_streamlit.txt과 fix_dependencies.py은 legacy파일로 옮겨도 무방.

2. 터미널에서 실행하는 명령어 : python main.py --query "인공지능 기술의 발전이 노동 시장에 미치는 영향을 분석하고, 이에 따른 윤리적 문제들과 정책적 대응 방안을 비교하여 설명하시오."

    -   쿼리는 예시

3. streamlit 실행 명령어:

    -   python run_streamlit.py
    -   streamlit run streamlit_app.py
        둘 다 가능

    -   streamlit 실행하면 화면이 뜰 때까지 시간이 꽤 소모. 터미널 창에서 어떤 RuntimeError가 두 개 뜰 떄까지 대기.
    -   RuntimeError가 뜨면 Streamlit 페이지 로딩 완료, RuntimeError는 무시해도 됨.(있어도 실행은 되더라구)

4. streamlit에서 쿼리넣고 실행시 총 두 번 langgraph 실행....ㅜ

    4-1) 첫 번째 모든 실행 마치면 executed step과 retrieved context를 출력 -> 바로 캡쳐, 두 번쨰 출력 나오면 지워짐.
   
    4-2) 두 번째 실행이 끝나면 최종 결과가 화면에 출력.

5. streamlit 실행 예시.

![Image](https://github.com/user-attachments/assets/12186f89-56ce-4d52-94d5-1f5a973a9b90)

![Image](https://github.com/user-attachments/assets/49749f8c-d467-44d4-8f3b-966b6aa526f5)
    
    -   execution plan : 맨 마지막 replan 과정에서 뽑아낸 plan인 듯.... 이거는 최대한 무시하는게 나을 듯.....?
    -   📊 Executed Steps (5 steps) : 전체 step이 어떤 plan을 따라갔는지 확인 가능. 하지만 그 plan에 대한 step 결과가 어떤 것인지까지는 출력이 안 나옴.
    -   지금 확인해보니 📊 Executed Steps (5 steps), 📚 Retrieved Context 이 두 부분 toggle하면 더 자세한 내용 볼 수 있도록 하는게 의도였는데 모든 출력이 반영이 안되었음...

![Image](https://github.com/user-attachments/assets/9aaa7e53-8805-4d2f-8866-01a8c22d1ac7)

    -   그래서 저 두 부분은 굳이 열지 말고 plan과 마지막 출력 결과물만 보여주는거에 신경쓰면 될 듯.
    -   시간이 있다면 고쳐볼텐데... 시간이 없네 ㅜ.......

5. 중요! img와 pdf 파일 첨부해서 첨부한 img나 pdf 파일에서 리트리벌 하는 기능 추가, streamlit으로도 img와 pdf 추가 가능
    -   img는 여러개 첨부 가능, pdf는 하나만 첨부 가능. pdf는 시간 엄청 오래걸림. 처리 과정이 길어서...
    -   이때 pdf 파일 처리를 위해서는 별도의 라이브러리를 수동으로 설치해야 함.
    -   https://m.blog.naver.com/chandong83/222262274082 이 블로그에서 환경변수 세팅하는 거까지만 따라하면 됨.
    -   환경변수 설정하면 내 경우에는 터미널 재부팅, 컴퓨터 재부팅까지 필요했음.

6. streamlit으로 실행 시 output 폴더에 graph_state_날짜.json으로 graph state가 저장되고 main_graph_, plan_execute_graph_로 그래프 시각화 이미지까지 저장.