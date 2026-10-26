import matplotlib.pyplot as plt
import numpy as np

models = ['TextRank', 'BiLSTM', 'IndicBART', 'mBART-50', 'Proposed']
rouge_scores = [41.2, 44.7, 48.5, 50.1, 55.6] 

# Plotting
plt.figure(figsize=(8,6))
bars = plt.bar(models, rouge_scores, color='#1f77b4')  

# Labels and title
plt.ylabel('ROUGE-1 (%)', fontsize=12)
plt.xlabel('Models', fontsize=12)
plt.title('ROUGE-1 Comparison', fontsize=14)

plt.tight_layout()
plt.show()

#--------------------------------

models = ['TextRank', 'BiLSTM', 'IndicBART', 'mBART-50', 'Proposed']
rouge2_scores = [18.5, 21.8, 24.7, 26.3, 31.2] 

# Plotting
plt.figure(figsize=(8,6))
bars = plt.bar(models, rouge2_scores, color='#1f77b4')  

# Labels and title
plt.ylabel('ROUGE-2 (%)', fontsize=12)
plt.xlabel('Models', fontsize=12)
plt.title('ROUGE-2 Comparison', fontsize=14)

plt.tight_layout()
plt.show()
#-------------------------------

models = ['TextRank', 'BiLSTM', 'IndicBART', 'mBART-50', 'Proposed']
rouge_l_scores = [37.5, 40.2, 44.1, 45.7, 50.8]  

plt.figure(figsize=(8,6))
bars = plt.bar(models, rouge_l_scores, color='#1f77b4')  

plt.title('ROUGE-L Comparison', fontsize=14)
plt.ylabel('ROUGE-L (%)', fontsize=12)
plt.xlabel('Models', fontsize=12)
plt.show()

#---------------------------------

models = ['TextRank', 'BiLSTM', 'IndicBART', 'mBART-50', 'Proposed']
bleu_scores = [21.3, 24.6, 27.8, 29.5, 34.8] 

plt.figure(figsize=(8,6))
plt.bar(models, bleu_scores, color='#1f77b4')
plt.title('BLEU Score Comparison', fontsize=14)
plt.ylabel('BLEU (%)', fontsize=12)
plt.xlabel('Models', fontsize=12)

plt.show()

#----------------------------

models = ['TextRank', 'BiLSTM', 'IndicBART', 'mBART-50', 'Proposed']
meteor_scores = [24, 28, 30.5, 32, 36.5] 

plt.figure(figsize=(8,6))
plt.bar(models, meteor_scores, color='#1f77b4') 

plt.title('METEOR Score Comparison', fontsize=14)
plt.ylabel('METEOR (%)', fontsize=12)
plt.xlabel('Models', fontsize=12)
plt.show()

#---------------------------------------
metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BLEU', 'METEOR']

ranks = {
    'TextRank': [5, 5, 5, 5, 5],
    'BiLSTM-Attention': [4, 4, 4, 4, 4],
    'IndicBART': [3, 3, 3, 3, 3],
    'mBART-50': [2, 2, 2, 2, 2],
    'Proposed Hybrid + TAMSORL': [1, 1, 1, 1, 1]
}

colors = {
    'TextRank': '#1f77b4',       
    'BiLSTM-Attention': '#ff7f0e', 
    'IndicBART': '#2ca02c',        
    'mBART-50': '#d62728',        
    'Proposed Hybrid + TAMSORL': '#9467bd' 
}

plt.figure(figsize=(8, 6))

for model, rank in ranks.items():
    plt.plot(metrics, rank, marker='o', label=model, color=colors[model], linewidth=2)

plt.gca().invert_yaxis()  
plt.xlabel('Evaluation Metrics', fontsize=12)
plt.ylabel('Rank (1 = Best)', fontsize=12)

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(loc='center right')
plt.show()

#----------------------------------

models = ['TextRank Only', 'TextRank + BiLSTM', 'without RL', 'Hybrid + TAMSORL']
rouge_l = [37.5, 43.0, 47.0, 51.0]  

plt.figure(figsize=(8,6))
bars = plt.bar(models, rouge_l, color='#1f77b4')

plt.ylabel('ROUGE-L (%)', fontsize=14)
plt.xlabel('Model Variants', fontsize=14,)


plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()
#-------------------------------------
episodes = np.arange(1, 51)  
average_reward = np.array([
-2.04761,
9.68531,
12.31467,
16.03918,
17.28166,
17.5751,
20.42357,
20.71691,
23.41937,
24.00473,
23.64095,
24.59151,
25.7609,
26.34616,
26.05567,
26.42221,
27.88352,
28.9068,
29.78417,
30.80765,
30.80879,
30.15347,
31.17675,
32.34623,
30.8144,
31.98426,
30.52485,
32.64347,
32.71781,
33.88758,
33.88891,
34.83919,
35.93557,
33.45469,
34.40506,
33.67664,
36.0869,
36.01542,
37.76893,
34.84974,
35.28938,
36.31285,
35.36542,
37.62948,
35.3677,
37.63224,
37.99868,
36.90486,
37.78233,
36.7618,

])

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(episodes, average_reward, marker='o', linestyle='-', color='#1f77b4')
plt.xlabel("Training Episodes", fontsize=12)
plt.ylabel("Average Reward", fontsize=12)
plt.grid(False)
plt.show()


