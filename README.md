<p align="center">
<img src="https://github-production-user-asset-6210df.s3.amazonaws.com/49296856/258254299-29c00dae-8201-4128-b341-dad4663b544a.jpg" width="400"> <br>
</p>


# [MM-Vet: Evaluating Large Multimodal Models for Integrated Capabilities](https://arxiv.org/abs/2308.02490)

<p align="center">
[<a href="https://arxiv.org/abs/2308.02490">Paper</a>] 
[<a href="https://github.com/yuweihao/MM-Vet/releases/download/v1/mm-vet.zip">Download dataset</a>]
[<a href="https://paperswithcode.com/sota/visual-question-answering-on-mm-vet">leaderboard</a>]
[<a href="https://huggingface.co/spaces/whyu/MM-Vet_Evaluator">Hugging Face Space</a>]
</p>

2023/12/23: :fire: :fire: We release inferences scripts for GPT-4V and Gemini. Gemini Pro Vision achieves 64.3% score, near 67.7% of GPT-4V.


2023/10/24 :fire: :fire: We evaluate GPT-4V on MM-Vet and observe that it achieves 67.7% score, outperforming other methods with large margin (20%). However, it still has a large gap to the full mark (100%), indicating the need for efforts to further improve the integrated capabilities of LMMs. See [leaderboard](https://paperswithcode.com/sota/visual-question-answering-on-mm-vet), [updated paper](https://arxiv.org/abs/2308.02490) and [GPT-4V prediction examples](#gpt-4v-prediction-examples).


2023/10/07 :fire: :fire: We released [MM-Vet leaderboard](https://paperswithcode.com/sota/visual-question-answering-on-mm-vet) on paperswithcode.com where you can add your model results conveniently. Note that date here means model date instead of paper date because some improved model versions are released after the paper.

In this repo, we offer data and evaluator of MM-Vet, proposed by our paper "[MM-Vet: Evaluating Large Multimodal Models for Integrated Capabilities](https://arxiv.org/abs/2308.02490)".


![MM-Vet](https://github-production-user-asset-6210df.s3.amazonaws.com/49296856/258206593-c61d21c8-cd9b-440c-9592-958d530bf122.png)
Figure 1: Different from conventional VL benchmarks only require one or two capabilities, MM-Vet focuses on the integration of different core VL capabilities, including recognition, OCR, knowledge, language generation, spatial awareness, and math.

## Evalute your model on MM-Vet
**Step 0**: Install openai package with `pip install openai>=1` and get access GPT-4/GPT-3.5 API. If you have not access, you can try MM-Vet online evaluator [Hugging Face Space](https://huggingface.co/spaces/whyu/MM-Vet_Evaluator) (but it may wait for long time depending on number of users).

**Step 1**:  Download MM-Vet data [here](https://github.com/yuweihao/MM-Vet/releases/download/v1/mm-vet.zip) and unzip `unzip mm-vet.zip`.

**Step 2**: Infer your model on MM-Vet and save your model outputs in json like [llava_llama2_13b_chat.json](results/llava_llama2_13b_chat.json), or just use [llava_llama2_13b_chat.json](results/llava_llama2_13b_chat.json) as example to evalute.

**Step 3**: `git clone https://github.com/yuweihao/MM-Vet.git && cd MM-Vet`, run LLM-based evaluator in [mm-vet_evaluator.ipynb](mm-vet_evaluator.ipynb) or [mm-vet_evaluator.py](mm-vet_evaluator.py) (Thanks to @HireTheHero to arrange it into py version).


## GPT-4V Prediction Examples
![GPT-4V prediction examples](https://github-production-user-asset-6210df.s3.amazonaws.com/15921929/277925013-daf61a18-5472-4064-88a1-d1c741bed018.png)

![GPT-4V prediction examples](https://github-production-user-asset-6210df.s3.amazonaws.com/15921929/277925959-269ff3fa-1f03-43b8-a497-60ee9e236eb3.png)

![GPT-4V prediction examples](https://github-production-user-asset-6210df.s3.amazonaws.com/15921929/277926586-0318ad3b-20b7-4995-b2bd-659e76b928fd.png)



## About running Bard
Please refer to these two files: [inference_bard.sh](misc/inference_bard.sh) and [inference_bard.py](misc/inference_bard.py).


## More samples on MM-Vet
![MM-Vet example 1](https://github-production-user-asset-6210df.s3.amazonaws.com/49296856/258215482-61d2e5d8-51c9-45ed-b35b-8e6c64bb26d8.jpg)

**Q**: What occasions would someone use this meme?

**GT**: This meme, commonly known as "Screaming Panda," is typically used to express shock, surprise, or fear. It could be used in response to a startling or unexpected event, or to convey a sense of panic or alarm. Some possible occasions where someone might use this meme include:

- Reacting to a jump scare in a horror movie
- Responding to a surprising plot twist in a TV show or book
- Expressing shock at a news headline or current event
- Conveying fear or anxiety about an upcoming deadline or exam
- Showing surprise at an unexpected outcome in a sports game or other competition.

**Required capabilities**: Recognition, knowledge, language generation

---

![MM-Vet example 2](https://github-production-user-asset-6210df.s3.amazonaws.com/49296856/258216998-2d3850e2-88cb-43e7-89c9-0e4cd50561fa.jpg)

**Q**: How many tomatoes are there?

**GT**: 5

**Required capabilities**: Recognition

---

![MM-Vet example 3](https://github-production-user-asset-6210df.s3.amazonaws.com/49296856/258217402-a9efcb58-d4e9-453d-8d9f-032c546c50fe.jpg)

**Q**: What is located to the right of the shampoo?

**GT**: conditioner

**Required capabilities**: OCR, spatial awareness

---

![MM-Vet example 4](https://github-production-user-asset-6210df.s3.amazonaws.com/49296856/258218056-c792ca0b-f0e9-4f1c-b6ea-b48ab5f1df0e.jpg)

**Q**: Which room is bigger, the double garage or the living room?

**GT**: double garage

**Required capabilities**: OCR, spatial awareness, math

---

![MM-Vet example 5](https://github-production-user-asset-6210df.s3.amazonaws.com/49296856/258218959-dea08352-d2ce-4b53-a201-899564c7ca73.jpg)

**Q**: On the right desk, what is to the left of the laptop?

**GT**: table lamp \<OR\> desk lamp

**Required capabilities**: Recognition, spatial awareness

---

![MM-Vet example 6](https://github-production-user-asset-6210df.s3.amazonaws.com/49296856/258219954-3581fa33-2a19-411c-93b3-b757515757be.jpg)

**Q**: What are all the scene text in the image?

**GT**: 5:30PM\<AND\>88%\<AND\>Mario Kart 8 Deluxe\<AND\>MARIO KART 8 DELUXE\<AND\>SUPER MARIO ODYSSEY\<AND\>THE LEGEND OF ZELDA\<AND\>BREATH OF WILD\<AND\>Options\<AND\>Start

**Required capabilities**: OCR

---

![MM-Vet example 7](https://github-production-user-asset-6210df.s3.amazonaws.com/49296856/258220686-66f92663-fdd0-49aa-9920-1d9e3d60f1cd.png)

**Q**: How many gallons of supreme gasoline can I get with $50?

**GT**: 13.6 \<OR\> 13.7

**Required capabilities**: OCR, math

---

![MM-Vet example 8](https://github-production-user-asset-6210df.s3.amazonaws.com/49296856/258221079-3836286e-2ba3-4c0f-b54d-f87ccd1a04b1.png)

**Q**: In which country was this photo taken?

**GT**: Australia

**Required capabilities**: Recognition, knowledge

---

![MM-Vet example 9](https://github-production-user-asset-6210df.s3.amazonaws.com/15921929/260642720-34a361ff-2350-494f-a557-4228700097c4.jpg)

**Q**: Can you explain this meme?

**GT**: This meme is a humorous take on procrastination and the tendency to delay tasks until a specific time. The person in the meme plans to do something at 8 o'clock, but when they miss that deadline by a few minutes, they decide to wait until 9 o'clock instead. The image of Kermit the Frog lying in bed represents the person's laziness and lack of motivation to complete the task.

**Required capabilities**: Recognition, OCR, knowledge, language generation

---

![MM-Vet example 10](https://github-production-user-asset-6210df.s3.amazonaws.com/49296856/258222130-16635f1c-6e68-4e13-83b1-86c3c5cafd03.png)

**Q**: The graph below shows the long-term international migration, UK, 1999-2008.

Summarize the information by selecting and reporting the main features, and make comparisons where relevant.

You should write at least 150 words.

**GT**: The chart gives information about UK immigration, emigration and net migration between 1999 and 2008.

Both immigration and emigration rates rose over the period shown, but the figures for immigration were significantly higher. Net migration peaked in 2004 and 2007.

In 1999, over 450,000 people came to live in the UK, while the number of people who emigrated stood at just under 300,000. The figure for net migration was around 160,000, and it remained at a similar level until 2003. From 1999 to 2004, the immigration rate rose by nearly 150,000 people, but there was a much smaller rise in emigration. Net migration peaked at almost 250,000 people in 2004.

After 2004, the rate of immigration remained high, but the number of people emigrating fluctuated. Emigration fell suddenly in 2007, before peaking at about 420,000 people in 2008. As a result, the net migration figure rose to around 240,000 in 2007, but fell back to around 160,000 in 2008.

**Required capabilities**: Recognition, OCR, language generation, spatial awareness

---

![MM-Vet example 11](https://github-production-user-asset-6210df.s3.amazonaws.com/49296856/258222558-65fa56c6-7721-43de-8938-9fba6f931291.jpg)

**Q**: Which car is on the parking spot 33?

**GT**: no \<OR\> empty

**Required capabilities**: Recognition, OCR, spatial awareness

---


![MM-Vet example 12](https://github-production-user-asset-6210df.s3.amazonaws.com/49296856/258222983-14a13e90-1907-4944-8617-30a3bb4f5feb.png)

**Q**: Is this apple organic?

**GT**: yes

**Required capabilities**: Recognition, OCR

---

![MM-Vet example 13](https://github-production-user-asset-6210df.s3.amazonaws.com/49296856/258223250-6fcde914-1541-45b9-9103-cc88c75a5408.png)

**Q**: Which are producers in this food web?

**GT**: Phytoplankton \<AND\> Seaweed

**Required capabilities**: OCR, knowledge, spatial awareness

---

![MM-Vet example 14](https://github-production-user-asset-6210df.s3.amazonaws.com/49296856/258223774-a05ddf4f-0d8e-4015-9c61-fd4bf09968f9.png)

**Q**: Is the person bigger than the car?

**GT**: no

**Required capabilities**: Recognition, knowledge, spatial awareness

---

![MM-Vet example 15](https://github.com/yuweihao/reclor/assets/49296856/75f61c60-d1d2-40c2-87fe-5633971cbe62)

**Q**: The table below gives information about the underground railway systems in six cities.

Summarise the information by selecting and reporting the main features, and make comparisons where relevant.

You should write at least 150 words.

**GT**: The table shows data about the underground rail networks in six major cities.

The table compares the six networks in terms of their age, size and the number of people who use them each year. It is clear that the three oldest underground systems are larger and serve significantly more passengers than the newer systems.

The London underground is the oldest system, having opened in 1863. It is also the largest system, with 394 kilometres of route. The second largest system, in Paris, is only about half the size of the London underground, with 199 kilometres of route. However, it serves more people per year. While only third in terms of size, the Tokyo system is easily the most used, with 1927 million passengers per year.

Of the three newer networks, the Washington DC underground is the most extensive, with 126 kilometres of route, compared to only 11 kilometres and 28 kilometres for the Kyoto and Los Angeles systems. The Los Angeles network is the newest, having opened in 2001, while the Kyoto network is the smallest and serves only 45 million passengers per year.

**Required capabilities**: OCR, language generation, spatial awareness

---

![MM-Vet example 16](https://github-production-user-asset-6210df.s3.amazonaws.com/49296856/258224681-852ab94c-e0da-4dc5-b080-fc3ca6311cfc.png)

**Q**: What will the girl on the right write on the board?

**GT**: 14

**Required capabilities**: Recognition, OCR, spatial awareness, math
