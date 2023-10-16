![img](./images/microsoft.jpg)

# Salary Data Classification

**Authors**: Franko Ndou

## Overview

The HR department plays a crucial role in identifying the right candidates for job positions. However, the process of determining a candidate's suitability can be time-consuming and subject to biases. To streamline their decision-making process and make it more data-driven, the HR department aims to develop a Machine Learning (ML) classification model. This model will predict whether a candidate is likely to earn more than $50,000 annually, helping the HR team make informed decisions about candidate qualifications.


## Business Problem

Microsoft is tasking me with demonstrating my data science expertise by assisting their HR department in developing a classification model. This model will predict whether job applicants are likely to earn more or less than $50,000 annually, helping Microsoft optimize its hiring process for a specific $50,000 annual salary position. By successfully developing this data-driven solution, I will not only contribute to improving their HR practices but also showcase my capabilities as a data scientist. Microsoft's commitment to innovation and efficiency in talent acquisition aligns with this project's goal.

![img](./images/giphy.gif)

## Data

I'm using salary cenus data that was donated on 4/30/1996 to the UCI Machine Learning Repository which can be found [here](https://archive.ics.uci.edu/dataset/20/census+income).

## Methods

Through upscaling a unbalanced dataset using SMOTE, using classes to test multiple hyperparameters, I was able to find the optimal Logistic Regression and Decision Tree model for my dataset.

## Baseline Models

Assembling a top-notch production team is essential for creating a successful film. Identifying the best director and writer for the job is crucial. While actors play significant roles, directors often craft roles with specific actors in mind. Therefore, determining the most successful actor may not directly contribute to our production team's ability to make the best possible movie. The success of a film largely hinges on the artistic vision of the director and the script quality. Relying solely on statistics related to actors may not enhance our return on investment (ROI) and could potentially have a detrimental impact on the film's quality.

### Logistic Regression

Baseline Matrix:

![img](./images/logregbase.png)

Baseline AUC:

![img](./images/logregbaseauc.png)

Scores:

Accuracy train Data: 82.82%
Accuracy test Data: 80.73%

Recall train Data: 85.93%
Recall test Data: 83.71%

Precision train Data: 80.9%
Precision test Data: 58.38%

F1 train Data: 83.34%
F1 test Data: 68.79%

The accuracy and AUC is already pretty good so it may be difficult to improve this model from its baseline. But the performance is defintiely looking good.

### Decision Tree

Baseline Decision Tree:
![img](./images/dtbase.png)

Baseline DT Matrix:
![img](./images/dtbasecon.png)

Baseline AUC:
![img](./images/dtbaseauc.png)

Scores:

Accuracy train Data: 98.38%
Accuracy test Data: 79.75%

Recall train Data: 97.69%
Recall test Data: 61.97%

Precision train Data: 99.05%
Precision test Data: 59.71%

F1 train Data: 98.37%
F1 test Data: 60.82%

This model is definitely overfitted which is to be expected with DT models. Its not performing as well as the LogReg either but we should base our decision on the tuned models. The AUC is fairly low compared to the LogReg as well.


## Trained Models

Action movies have the highest ROI among different genres by a significant margin. This aligns perfectly with the goal of working with a director who excels in this genre.

```python
# Create a bar plot for average ROI by genre
plt.figure(figsize=(12, 6))
sns.barplot(data=genre_roi_avg, x='genre', y='ROI', palette='viridis')
plt.xticks(rotation=90)
plt.xlabel('Genre')
plt.ylabel('Average ROI')
plt.title('Average ROI by Genre')
plt.tight_layout()
plt.show()
```
![img](./images/avgroigenre.png)

### Logistic Regression

Wednesday is the best day to release our movie, with the highest ROI. Releasing during the warmer months, such as June, July, and May, is recommended as people are more likely to go to the movies during this period. This hypothesis likely supports the larger ROI's during those months.

```python
# Create bars for day of the week vs. domestic gross
sns.barplot(x='day_of_the_week', y='domestic_gross_in_mill', data=budgets)
sns.set(font_scale=0.75)
plt.ylabel('Domestic Gross per million')
plt.title('Domestic Gross on Given Day of the Week')
plt.show()

# Create bars for month of the year vs. domestic gross
sns.barplot(x='month_of_the_year', y='domestic_gross_in_mill', data=budgets)
sns.set(font_scale=0.75)
plt.xticks(rotation=45)
plt.ylabel('Domestic Gross per million')
plt.title('Domestic Gross by Month of the Year')
plt.show()
```

![img](./images/bestdayforfilm.png)
![img](./images/month.png)

### Decision Tree

Selecting a talented musician with a track record of working on successful films is crucial for creating a captivating soundtrack that enhances the overall impact and success of our movie.

```python
# Merge two DataFrames 'merged_df' and 'people_and_movies_df' using an outer join
relevant_people_and_movies = pd.merge(merged_df, people_and_movies_df, how='outer', left_on='movie', right_on='original_title')

# Drop columns we don't need
relevant_people_and_movies = relevant_people_and_movies.drop(['release_date',
  'runtime_minutes', 'original_title', 'genre'], axis=1)

# Sort the DataFrame by 'ROI' in descending order
relevant_people_and_movies = relevant_people_and_movies.sort_values(by='ROI', ascending=False)

# Drop rows with missing 'primary_profession' values
relevant_people_and_movies = relevant_people_and_movies.dropna(subset=['primary_profession'])

# Filter the DataFrame based on primary profession
chosen_artists = relevant_people_and_movies[relevant_people_and_movies['primary_profession'].str.contains \
                                                  ('soundtrack|composer|music_department|sound_deparment')]
# Select relevant columns
chosen_artists = chosen_artists[['primary_name', 'primary_profession', 'ROI', 'averagerating', 'numvotes']]

# Drop duplicate rows, if any
chosen_artists = chosen_artists.drop_duplicates()

# Sort by 'average rating' in descending order 
chosen_artists = chosen_artists.sort_values(by='averagerating', ascending=False)

# Chose only 'successful' artists by setting the roi to 2 as well as the minimum rating to 7 and display final result
chosen_artists = chosen_artists[chosen_artists['ROI'] > 1.74]
chosen_artists = chosen_artists[chosen_artists['numvotes'] > 10000]
chosen_artists[chosen_artists['averagerating'] >= 7]
```

![img](./images/sound_team.png)



## Conclusion

An EDA only allows us to look at the statistical data and come up with likely probabilities. There is no way to guarantee the performance of any individual director, actor, musician, or genre. However, with that being said, we feel comfortable creating business recommendations based on the likelihood of these events occurring, as well as basing them on the trends within the industry.

- Categorically, action films generate the largest return on investment by far, compared to any other movie genre. This is likely due to the mass popularity of action films. We should aim to create an action movie as it will have the largest target audience as well as generate the most ROI for our company.

- The best day to release our film is during Wednesday. Most films are actually released on Friday; however, Wednesday has the largest ROI by far. We should aim for the warmer months as well, such as June, July, and May. As those months tend to generate the most ROI as well.
    - This is likely because during the warmer months, people are willing to drive out to see movies and spend time, whereas during colder months, people will tend to stay home to avoid inclement weather.
    - Wednesday likely generates the highest ROI because it's in the middle of the week, which allows most people to view the film within the first week. However, there is no real way to prove this theory; all we know is that Wednesday generates the highest ROI on average.

- We should absolutely work with a talented director who is not only critically acclaimed but also has a reputation for generating a positive ROI. Through data exploration as well as visualizations, we've come to the conclusion that the current best option would be Christopher Nolan. We have also provided a table of potential backup directors who all meet our criteria, assuming Mr. Nolan is not available.
    - Mr. Nolan generates an above-average ROI compared to other high-budget films (1.74 Avg / 3.13 Nolan).
    - He has one of the highest IMDb user rating scores (8.8).
    - The majority of his movies fall within a 7.8-8.8 range of user ratings.
    - The probability of a Christopher Nolan film underperforming is close to 3.8%.

- The score of a movie is incredibly important as it often dictates the mood and ambiance of the film. With that being said, we have provided a list of potential musicians who fall within our selected criteria.
    - Sam Estes and Andrew Kawczynski are our two recommendations for soundtrack producers. They have collectively worked on incredible movies such as Inception, Planet of the Apes, Interstellar, Top Gun Maverick and The Dark Knight.
        
## Next Steps

To provide even more insight for Universal Pictures, these are the steps we could take

- We can build predictive models using machine learning techniques. We could use features like director, genre, release date, and budget to predict box office performance. This would require splitting our data into training and testing sets, selecting appropriate algorithms, and evaluating model performance.

- Import a database that has a detailed filmography of directors and studios, We lost a some data from cleaning due to the fact it was incredibly difficult to filter and maintain everything that was relevant. Some newer movies are not on this list either, and it would be important to analyze the current market rather than the market a few years ago.

- Compare how best-sellers in all film genres compete with each other. Action movies may have the highest ROI on average but we should look at the outliers for each genre and see how popular they became and how much they generated.

- Research how musical data influences the success of films. What musical genres have the highest impact on other film genres? What is the average length of a song in each genre, what variety of musical genres are within an OST? 

- Compare and contrast ROI and user rating relative to genre for directors. See how directors perform in each individual genre and compare it to the averages of other directors and genres.

## For More Information

Please review the full analysis in [my Jupyter Notebook](./code/Salary-Classifcation-Models.ipynb) or my [presentation](./Salary_Prediction.pdf).

For any additional questions, please contact me:

**Franko Ndou & frankondou@gmail.com**

## Repository Structure

Describe the structure of your repository and its contents, for example:

```
├── code
│   ├── __init__.py
│   ├── Cleaning-Salary-Data.ipynb
│   └── Salary-Classifcation-Models.ipynb
├── images
├── README.md
├── .gitignore
└── Salary_Prediction.pdf
```