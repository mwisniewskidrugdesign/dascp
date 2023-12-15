# -*- coding: utf-8 -*-
# Nie opisałem jeszcze procesu instalacji pakietu matplotlib,
# a więc możesz oznaczyć poniższą linię symbolem komentarza.
from matplotlib import pyplot as plt

#####################################
#                                   #
# Określanie najważniejszych węzłów #
#                                   #
#####################################

users = [
    { "id": 0, "name": "Hero" },
    { "id": 1, "name": "Dunn" },
    { "id": 2, "name": "Sue" },
    { "id": 3, "name": "Chi" },
    { "id": 4, "name": "Thor" },
    { "id": 5, "name": "Clive" },
    { "id": 6, "name": "Hicks" },
    { "id": 7, "name": "Devin" },
    { "id": 8, "name": "Kate" },
    { "id": 9, "name": "Klein" },
    { "id": 10, "name": "Jen" }
]

friendships = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4),
               (4, 5), (5, 6), (5, 7), (6, 8), (7, 8), (8, 9)]


# Zacznij od utworzenia pustej listy dla każdego użytkownika.
for user in users:
    user["friends"] = []

# Teraz wypełnij te listy danymi.
for i, j in friendships:
    # Kod ten działa, ponieważ użytkownik users[i], to użytkownik o identyfikatorze i.
    users[i]["friends"].append(users[j]) # dodaje i do listy przyjaciół j
    users[j]["friends"].append(users[i]) # dodaje j do listy przyjaciół i
	
def number_of_friends(user):
    """Ile znajomuych na użytkownik?"""
    return len(user["friends"]) # długość listy friend_ids

total_connections = sum(number_of_friends(user)
                        for user in users) # 24

num_users = len(users)
avg_connections = total_connections / num_users # 2,4

##################################
#                                #
# Analitycy, których możesz znać #
#                                #
##################################

def friends_of_friend_ids_bad(user):
    # Przyjaciół przyjaciół oznaczamy akronimem foaf.
    return [foaf["id"]
            for friend in user["friends"] # Dla każdego przyjaciela
            for foaf in friend["friends"]] # odczytaj listę jego przyjaciół.

from collections import Counter # Pakiet ten nie jest ładowany domyślnie.

def not_the_same(user, other_user):
    """Użytkownicy other_user muszą posiadać różne identyfikatory"""
    return user["id"] != other_user["id"]

def not_friends(user, other_user):
    """Użytkownik other_user nie jest znajomym, jeżeli nie należy do listy user["friends"]."""
    return all(not_the_same(friend, other_user)
               for friend in user["friends"])

def friends_of_friend_ids(user):
    return Counter(foaf["id"]
                   for friend in user["friends"]  # Dla każdego z moich przyjaciół
                   for foaf in friend["friends"]  # wybierz ich znajomych,
                   if not_the_same(user, foaf)    # którzy nie są mną
                   and not_friends(user, foaf))   # i nie są moimi znajomymi.

print(friends_of_friend_ids(users[3])) # licznik ({0: 2, 5: 1})

interests = [
    (0, "Hadoop"), (0, "Big Data"), (0, "HBase"), (0, "Java"),
    (0, "Spark"), (0, "Storm"), (0, "Cassandra"),
    (1, "NoSQL"), (1, "MongoDB"), (1, "Cassandra"), (1, "HBase"),
    (1, "Postgres"), (2, "Python"), (2, "scikit-learn"), (2, "scipy"),
    (2, "numpy"), (2, "statsmodels"), (2, "pandas"), (3, "R"), (3, "Python"),
    (3, "statistics"), (3, "regression"), (3, "probability"),
    (4, "machine learning"), (4, "regression"), (4, "decision trees"),
    (4, "libsvm"), (5, "Python"), (5, "R"), (5, "Java"), (5, "C++"),
    (5, "Haskell"), (5, "programming languages"), (6, "statistics"),
    (6, "probability"), (6, "mathematics"), (6, "theory"),
    (7, "machine learning"), (7, "scikit-learn"), (7, "Mahout"),
    (7, "neural networks"), (8, "neural networks"), (8, "deep learning"),
    (8, "Big Data"), (8, "artificial intelligence"), (9, "Hadoop"),
    (9, "Java"), (9, "MapReduce"), (9, "Big Data")
]

def data_scientists_who_like(target_interest):
    return [user_id
            for user_id, user_interest in interests
            if user_interest == target_interest]

from collections import defaultdict

# Klucze są zainteresowaniami, wartości są listami identyfikatorów użytkowników interesujących się danym tematem.
user_ids_by_interest = defaultdict(list)

for user_id, interest in interests:
    user_ids_by_interest[interest].append(user_id)

# Klucze są identyfikatorami użytkowników, wartości są listami zainteresowań danego użytkownika.
interests_by_user_id = defaultdict(list)

for user_id, interest in interests:
    interests_by_user_id[user_id].append(interest)

def most_common_interests_with(user_id):
    return Counter(interested_user_id
        for interest in interests_by_user_id["user_id"]
        for interested_user_id in user_ids_by_interest[interest]
        if interested_user_id != user_id)

#################################
#                               #
# Wynagrodzenie i doświadczenie #
#                               #
#################################

salaries_and_tenures = [(83000, 8.7), (88000, 8.1),
                        (48000, 0.7), (76000, 6),
                        (69000, 6.5), (76000, 7.5),
                        (60000, 2.5), (83000, 10),
                        (48000, 1.9), (63000, 4.2)]

def make_chart_salaries_by_tenure():
    tenures = [tenure for salary, tenure in salaries_and_tenures]
    salaries = [salary for salary, tenure in salaries_and_tenures]
    plt.scatter(tenures, salaries)
    plt.xlabel("Lata pracy")
    plt.ylabel("Zarobki")
    plt.show()

# Klucze są latami, wartości określają wynagrodzenie dla danego stażu pracy.
salary_by_tenure = defaultdict(list)

for salary, tenure in salaries_and_tenures:
    salary_by_tenure[tenure].append(salary)

average_salary_by_tenure = {
    tenure : sum(salaries) / len(salaries)
    for tenure, salaries in salary_by_tenure.items()
}

def tenure_bucket(tenure):
    if tenure < 2: return "mniej niż dwa"
    elif tenure < 5: return "pomiędzy dwoma i pięcioma"
    else: return "powyżej pięciu"

salary_by_tenure_bucket = defaultdict(list)

for salary, tenure in salaries_and_tenures:
    bucket = tenure_bucket(tenure)
    salary_by_tenure_bucket[bucket].append(salary)

average_salary_by_bucket = {
  tenure_bucket : sum(salaries) / len(salaries)
  for tenure_bucket, salaries in salary_by_tenure_bucket.items()
}


#################
#               #
# Płatne konta  #
#               #
#################

def predict_paid_or_unpaid(years_experience):
  if years_experience < 3.0: return "płatne"
  elif years_experience < 8.5: return "darmowe"
  else: return "płatne"

####################################
#                                  #
# Tematy interesujące użytkowników #
#                                  #
####################################

words_and_counts = Counter(word
                           for user, interest in interests
                           for word in interest.lower().split())


if __name__ == "__main__":

    print()
    print "###################################"
    print "#"
    print "# Określanie najważniejszych węzłów"
    print "#"
    print "###################################"
    print()


    print "suma znajomości", total_connections
    print "liczba użytkowników", num_users
    print "średnia liczba znajomości", total_connections / num_users
    print()

    # Zacznij od utworzenia pustej listy dla każdego użytkownika
    num_friends_by_id = [(user["id"], number_of_friends(user))
                         for user in users]

    print "Użytkownicy posortowani według liczby znajomych"
    print(sorted(num_friends_by_id,
                 key=lambda pair: pair[1],                       # Sortuj według liczby znajomych
                 reverse=True))                                  # od najwyższej do najniższej

    print()
    print "################################"
    print "#"
    print "# Analitycy, których możesz znać"
    print "#"
    print "################################"
    print()


    print "Znajomi znajomych użytkownika 0:", friends_of_friend_ids_bad(users[0])
    print "Znajomi znajomych użytkownika 3:", friends_of_friend_ids(users[3])

    print()
    print "###############################"
    print "#"
    print "# Wynagrodzenie i doświadczenie"
    print "#"
    print "###############################"
    print()

    print "Średnie wynagrodzenie a staż pracy", average_salary_by_tenure
    print "Średnie wynagrodzenie a grupy stażu pracy", average_salary_by_bucket

    print()
    print "#########################"
    print "#"
    print "# Najpopularniejsze słowa"
    print "#"
    print "#########################"
    print()

    for word, count in words_and_counts.most_common():
        if count > 1:
            print(word, count)
