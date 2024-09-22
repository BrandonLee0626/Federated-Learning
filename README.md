# Federated-Learning

#Todo
1. FedAvg 수식 이해하기
    수식에서 loss function의 parameter가 되는 model은 update된 model
2. 이해한 수식 기반으로 client랑 server 구현하기
3. train이 완료되고 weight가 어떤 type으로 나오는지 파악 필요

layer가 3개인 model -> 각 layer 별로 weight랑 bias가 있을 것
그럼 FedAvg를 각 layer의 weight별로 진행, bias별로 진행

참고자료

Federated Learning 기본틀: https://towardsdatascience.com/federated-learning-a-step-by-step-implementation-in-tensorflow-aac568283399

PyTorch로 MNIST 분류 모델 구현: https://velog.io/@gr8alex/PyTorch%EB%A5%BC-%EC%9D%B4%EC%9A%A9%ED%95%9C-MNIST-%EB%B6%84%EB%A5%98-%EB%AA%A8%EB%8D%B8-%EB%A7%8C%EB%93%A4%EC%96%B4%EB%B3%B4%EA%B8%B0