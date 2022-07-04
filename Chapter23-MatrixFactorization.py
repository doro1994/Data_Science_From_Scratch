from scratch.linear_algebra import dot
from typing import List
from scratch.deep_learning import random_tensor
from typing import NamedTuple
import csv
import re
import random
import tqdm
from collections import defaultdict

MOVIES = "./data/ml-100k/u.item"
RATINGS = "./data/ml-100k/u.data"

class Rating(NamedTuple):
    user_id: str
    movie_id:str
    rating: float
    
with open(MOVIES, encoding="iso-8859-1") as f:
    reader = csv.reader(f, delimiter="|")
    movies = {movie_id: title for movie_id, title, *_ in reader}
    
with open(RATINGS, encoding="iso-8859-1") as f:
    reader = csv.reader(f, delimiter="\t")
    ratings = [Rating(user_id, movie_id, float(rating))
                for user_id, movie_id, rating, _ in reader]
    
star_wars_ratings = {movie_id: []
                    for movie_id, title in movies.items()
                    if re.search("Star Wars|Empire Strikes|Jedi", title)}

for rating in ratings:
    if rating.movie_id in star_wars_ratings:
        star_wars_ratings[rating.movie_id].append(rating.rating)
        
avg_ratings = [(sum(title_ratings) / len(title_ratings), movie_id)
                for movie_id, title_ratings in star_wars_ratings.items()]

for avg_rating, movie_id in sorted(avg_ratings, reverse=True):
    print(f"{avg_rating:.2f} {movies[movie_id]}")

random.seed(0)
random.shuffle(ratings)

split1 = int(len(ratings) * 0.7)
split2 = int(len(ratings) * 0.85)

train = ratings[:split1]
validation = ratings[split1:split2]
test = ratings[split2:]

avg_rating = sum(rating.rating for rating in train) / len(train)
baseline_error = sum((rating.rating - avg_rating) ** 2
                     for rating in test) / len(test)

EMBEDDING_DIM = 2
user_ids = {rating.user_id for rating in ratings}
movie_ids = {rating.movie_id for rating in ratings}

user_vectors = {user_id: random_tensor(EMBEDDING_DIM)
                for user_id in user_ids}
movie_vectors = {movie_id: random_tensor(EMBEDDING_DIM)
                for movie_id in movie_ids}

def loop(dataset: List[Rating],
         learning_rate: float = None) -> None:
    with tqdm.tqdm(dataset) as t:
        loss = 0.0
        for i, rating in enumerate(t):
            movie_vector = movie_vectors[rating.movie_id]
            user_vector = user_vectors[rating.user_id]
            predicted = dot(user_vector, movie_vector)
            error = predicted - rating.rating
            loss += error ** 2
            
            if learning_rate is not None:
                user_gradient = [error * m_j for m_j in movie_vector]
                movie_gradient = [error * u_j for u_j in user_vector]
                
                for j in range(EMBEDDING_DIM):
                    user_vector[j] -= learning_rate * user_gradient[j]
                    movie_vector[j] -= learning_rate * movie_gradient[j]
            t.set_description(f"avg loss: {loss / (i + 1)}")    
            
learning_rate = 0.05
for epoch in range(20):
    learning_rate *= 0.9
    print(epoch, learning_rate)
    loop(train, learning_rate=learning_rate)
    loop(validation)
loop(test)            

from scratch.working_with_data import pca, transform

original_vectors = [vector for vector in movie_vectors.values()]
components = pca(original_vectors, 2)

ratings_by_movie = defaultdict(list)
for rating in ratings:
    ratings_by_movie[rating.movie_id].append(rating.rating)
    
vectors = [
        (movie_id,
         sum(ratings_by_movie[movie_id]) / len(ratings_by_movie[movie_id]),
         movies[movie_id],
         vector)
        for movied_id, vector in zip(movie_vectors.keys(),
                                     transform(original_vectors, components))
        ]

print(sorted(vectors, key=lambda v: v[-1][0])[:25])
print(sorted(vectors, key=lambda v: v[-1][0])[-25:])

            
