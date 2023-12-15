from matplotlib import pyplot as plt

years = [1950, 1960, 1970, 1980, 1990, 2000, 2010]
gdp = [300.2, 543.3, 1075.9, 2862.5, 5979.6, 10289.7, 14958.3]

# Utwórz wykres liniowy, przedstaw lata na osi x, a PKB na osi y.
plt.plot(years, gdp, color='green', marker='o', linestyle='solid')

# Dodaj tytuł.
plt.title("Nominal GDP")

# Dodaj tytuł osi y.
plt.ylabel("Billions of $")
plt.show()


plt.savefig('im/viz_gdp.png')
plt.gca().clear()

movies = ["Annie Hall", "Ben-Hur", "Casablanca", "Gandhi", "West Side Story"]
num_oscars = [5, 11, 3, 8, 10]

# Narysuj słupki o współrzędnych x [0, 1, 2, 3, 4] i wysokości [num_oscars].
plt.bar(range(len(movies)), num_oscars)

plt.title("My Favorite Movies")     # Dodaj tytuł
plt.ylabel("# of Academy Awards")   # Dodaj etykietę osi y

# Na osi x nanieś etykietę w postaci tytułów filmów i je wyśrodkuj.
plt.xticks(range(len(movies)), movies)

plt.show()


plt.savefig('im/viz_movies.png')
plt.gca().clear()

from collections import Counter
grades = [83, 95, 91, 87, 70, 0, 85, 82, 100, 67, 73, 77, 0]

# podziel stopnie na decyle, ale tak, żeby wartość 100 znalazła się w jednym przedziale z 91 i 95
histogram = Counter(min(grade // 10 * 10, 90) for grade in grades)

plt.bar([x + 5 for x in histogram.keys()],  # Przesuń słupki w prawo o 5.
        histogram.values(),                 # Nadaj każdemu słupkowi właściwą wysokość.
        10,                                 # Nadaj każdemu słupkowi szerokość 10.
        edgecolor=(0, 0, 0))                # Czarny obrys każdego słupka.

plt.axis([-5, 105, 0, 5])                  # Zdefiniuj zakres osi x od –5 do 105
                                           # i zakres osi y od 0 do 5.

plt.xticks([10 * i for i in range(11)])    # Umieść etykiety osi x w punktach 0, 10, ..., 100.
plt.xlabel("Decyl")
plt.ylabel("Liczba studentow")
plt.title("Rozklad ocen z pierwszego egzaminu")
plt.show()


plt.savefig('im/viz_grades.png')
plt.gca().clear()

mentions = [500, 505]
years = [2017, 2018]

plt.bar(years, mentions, 0.8)
plt.xticks(years)
plt.ylabel("Ile razy uslyszalem fraze nauka o danych?")

# Jeżeli tego nie zrobisz, to matplotlib umieści na osi x etykiety 0 i 1,
# a następnie doda w rogu +2.013e3 (niegrzeczny matplotlib!).
plt.ticklabel_format(useOffset=False)

# W celu zamazania obrazu sytuacji oś y zaczyna się od punktu 500.
plt.axis([2016.5, 2018.5, 499, 506])
plt.title("Patrz na ten ogromny wzrost!")
plt.show()


plt.savefig('im/viz_misleading_y_axis.png')
plt.gca().clear()


plt.bar(years, mentions, 0.8)
plt.xticks(years)
plt.ylabel("Ile razy uslyszalem fraze nauka o danych?")
plt.ticklabel_format(useOffset=False)

plt.axis([2016.5, 2018.5, 0, 550])
plt.title("Niewielki wzrost")
plt.show()


plt.savefig('im/viz_non_misleading_y_axis.png')
plt.gca().clear()

variance     = [1, 2, 4, 8, 16, 32, 64, 128, 256]
bias_squared = [256, 128, 64, 32, 16, 8, 4, 2, 1]
total_error  = [x + y for x, y in zip(variance, bias_squared)]
xs = [i for i, _ in enumerate(variance)]

# Funkcja plt.plot może być wywoływana wielokrotnie
# w celu umieszczenia wielu serii danych na tym samym wykresie.
plt.plot(xs, variance,     'g-',  label='variance')    # zielona linia ciągła
plt.plot(xs, bias_squared, 'r-.', label='bias^2')      # czerwona linia składająca się z kropek i kresek
plt.plot(xs, total_error,  'b:',  label='total error') # niebieska linia punktowana

# Przypisaliśmy etykiety do wszystkich serii danych,
# a więc legenda zostanie wygenerowana automatycznie.
# Parametr loc=9 umieszcza ją na środku u góry wykresu.
plt.legend(loc=9)
plt.xlabel("Stopien skomplikowania modelu")
plt.xticks([])
plt.title("Kompromis pomiędzy wartoscia progowa i zlozonoscia modelu")
plt.show()


plt.savefig('im/viz_line_chart.png')
plt.gca().clear()

friends = [ 70,  65,  72,  63,  71,  64,  60,  64,  67]
minutes = [175, 170, 205, 120, 220, 130, 105, 145, 190]
labels =  ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

plt.scatter(friends, minutes)

# Nadaj etykiety wszystkim punktom.
for label, friend_count, minute_count in zip(labels, friends, minutes):
    plt.annotate(label,
        xy=(friend_count, minute_count), # Umieść etykietę we właściwym miejscu,
        xytext=(5, -5),                  # ale lekko ją przesuń.
        textcoords='offset points')

plt.title("DCzas spedzony na stronie a liczba znajomych")
plt.xlabel("Liczba znajomych")
plt.ylabel("Dzienny czas spedzony na stronie (minuty)")
plt.show()


plt.savefig('im/viz_scatterplot.png')
plt.gca().clear()

test_1_grades = [ 99, 90, 85, 97, 80]
test_2_grades = [100, 85, 60, 90, 70]

plt.scatter(test_1_grades, test_2_grades)
plt.title("Osie nie sa porownywalne")
plt.xlabel("test 1")
plt.ylabel("test 2")
# plt.show()


plt.savefig('im/viz_scatterplot_axes_not_comparable.png')
plt.gca().clear()


test_1_grades = [ 99, 90, 85, 97, 80]
test_2_grades = [100, 85, 60, 90, 70]
plt.scatter(test_1_grades, test_2_grades)
plt.title("Osie sa porownywalne")
plt.axis("równe")
plt.xlabel("test 1")
plt.ylabel("test 2")
plt.savefig('im/viz_scatterplot_axes_comparable.png')
plt.gca().clear()

