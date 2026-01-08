논문 서론 서술에서, 기존 continual learning과 continual ad와의 task 관점에서 고유 차이점 어느정도 서술 필요 

- continual learning : 기존에 성능은 어느정도였는데, forgetting으로 인해 성능이 저하되는 것을 완화 시킨다. 
- continual ad : 기존 방법론과는 독자적인 방법론을 대체로 구축후 continual learning 시나리오에서 강건한 성능을 보여주도록 함 

이럴 때 parameter isolation을 조금 더 강조해서 풀어나가면 어떨지 


tail aware loss에서 ratio 0.02에 대한 합당한 근거 및 실험 필요 
- 단순히 0.02 비율의 likelihood들에 대해 weight를 주었을 때 이렇게 까지 성능 차이가 나는지에 대한 분석 필요 
- 반대로 0.05 로 커지는 경우 반대로 성능이 크게 저하 : 이에 대한 가설로, tail의 너무 많은 부분을 weight를 주게 되면, 이 tail들이 gaussian distribution의 head가 되고, 정작 가장 많은 확률 및 분포를 차지하는 특징들이 tail로 이동 됨 -> 이 현상이 반복되면서, 정상적으로 gaussian distribution을 학습하지 못 하고 뒤틀어짐 -> 이걸 보여주는 실험 필요 