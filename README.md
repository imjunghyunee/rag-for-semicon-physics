# rag-for-semicon-physics

Further modifications necessary:
1. Query Decomposition
   When query is considered "complex", initiate "query_decomposition"
   
  ````
  Query decomposition means:
  1-1. Based on the original query, llm generates subquestions to solve the complex query question.
  1-2. Retrieval process is done for each subquestion. (max_subquestion_num=5)
  1-3. Add all contexts and answer for each subquestion.
  1-4. Based on that as context, llm finally ouputs final answer.
````

2. Modify embedding model fine-tuning code.
  Find possible reason for the code "./train/train_jina_hard_neg_mining_final.py" to fail training the model properly
  "./train/EmbbeddingsFtRAG" directory의 fine-tuning 코드 참고해서 fine-tuning 코드 재작성

Notes.
이거 llm 세팅이 localhost에 띄워놓은 llm 기준으로 되어있어서....
잘 작동되는지 테스트해보려면 OpenAI 세팅으로 코드 바꿔서 테스트해야 됨
