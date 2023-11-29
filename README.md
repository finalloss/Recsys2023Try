# Recsys2023Try
For Recsys 2023, private projects -> public projects

Recsys Challenge 2023 is a competition in the field of recommendation systems, aiming to explore the performance of recommendation algorithms on large-scale data sets. Provide innovative solutions for research and applications in the field of recommendation systems. This article aims to detail our participation in the Recsys Challenge The work on the 2023 data classification task includes problem description, problem analysis, experimental plan, experimental process, and final results and analysis.

# Brief introduction

## [About](http://www.recsyschallenge.com/2023/#top)

The RecSys 2023 Challenge will be organized by Sarang Brahme, Rahul Agarwal (ShareChat), Abhishek Srivastava (IIM Visakhapatnam, India), Liu Yong (Huawei, Singapore) and Athirai Irissappane (Amazon, USA) based on the data provided by ShareChat. This year’s challenge will focus on online advertising, improving deep funnel optimization, and user privacy.

The challenge is brought to you by ShareChat. [ShareChat](https://sharechat.com/about) is India’s largest homegrown social media company, with 400+ million MAUs across all its platforms. Headquartered in Bengaluru, ShareChat is spreading its team globally across India, the USA, and Europe. We have the best-in-class AI & ML technology and the strongest feed ranking system powering our growth. We aim to create a million monetizable creators with USD 450 million in creator earnings across ShareChat and Moj by 2025.

## [Challenge Task](http://www.recsyschallenge.com/2023/#top)

Online advertising has been a multi-billion dollar industry since the early 2000 and has played a significant role in the growth of the internet. The key advantage of online advertising over conventional mass advertising is its inherent ability to personalize to users, democratizing advertising and enabling businesses of all sizes to participate, and providing the measurable impact of money spent to the advertisers. Over the past two decades, the nature of online advertising has also evolved tremendously from pure banner-based advertising, where advertisers were charged based on the number of ad impressions, to deep funnel optimizations, where advertisers can optimize for eventual sales.

The efficacy of deep funnel optimization required extensive personalization and opened up rich problems in real-time auction design, large-scale machine learning, modeling delayed feedback, and behavioral understanding. As these systems matured, we also started developing a rich understanding of the need to preserve user privacy, ensure AI fairness, and prevent adversarial exploitation of the platform. In this challenge, we aim to provide a real-world ad dataset from the Sharechat and Moj apps to act as a benchmark for research into deep funnel optimization with a focus on user privacy

## [DataSet](http://www.recsyschallenge.com/2023/#top)

1. The dataset corresponds to roughly 10M random users who visited the ShareChat + Moj app over three months. We have sampled each user's activity to generate **10 impressions** corresponding to each user. Our target variable is whether there was an install for an app by the user or not.
2. To represent a user, several features are provided:
   1. **Demographic features**: These include age, gender, and geographic location from where the user is accessing the Sharechat/Moj app. The sampling of the users in (1) is done such that we have an approximately uniform distribution of users across the demographic features. The user's location is hashed to a 32-bit to anonymize the data.
   2. **Content preference embeddings**: These embeddings are trained based on the users' consumption of the various non-ad content on the Sharechat/Moj app.
   3. **App affinity embeddings**: These embeddings are trained based on the past apps installed by the user on our platform.
3. We also have features corresponding to ads
   1. **Ad categorical features**: These features represent different characteristics of an ad, including the size of the ad, the category of the ad etc. The features are hashed to 32-bit to anonymize the data
   2. **Ad embedding**: These represent the actual video/image content of the ad.
4. To capture the historical interactions between users and ads, we also provide
   1. **Count features**: These features represent the user interaction with ads, advertisers, and categories of advertisers over different lengths of a time window
5. Every row of the data has an associated numeric id and represents an ad impression shown to the user and whether it resulted in a click on the ad and subsequently an install or not.
6. We do not provide the semantics of the individual features.
7. The training data consists of subsampled impressions/clicks/installs from the past 2 weeks and aims to predict the probability of install for the 15th day.

## [Prize](http://www.recsyschallenge.com/2023/#top)

- First three teams from the participants - $2500/$1500/$1000
- Special prize for the academic teams - $1500

# [Timeline](http://www.recsyschallenge.com/2023/#top)

| When?                         | What?                                                        |
| :---------------------------- | :----------------------------------------------------------- |
| 27 March, 2023                | **Start RecSys Challenge**Release dataset                    |
| 11 Apr, 2023                  | **Submission System Open**                                   |
| 13 Apr, 2023                  | **Leaderboard live**                                         |
| 22nd June, 2023 18 June, 2023 | **End RecSys Challenge**                                     |
| 28th June, 2023 24 June, 2023 | **Final Leaderboard & Winners**EasyChair open for submissions |
| 3rd July, 2023 30 June, 2023  | **Code Upload**Upload code of the final predictions          |
| 14 July, 2023                 | **Paper Submission Due**                                     |
| 1 August, 2023                | **Paper Acceptance Notifications**                           |
| 14 August, 2023               | **Camera-Ready Papers**                                      |
| Sept 3rd week                 | **RecSys Challenge Workshop**@ [ACM RecSys 2023](https://recsys.acm.org/recsys23/) |
