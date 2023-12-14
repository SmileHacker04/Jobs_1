import requests
import tkinter as tk
from PIL import Image, ImageTk
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from tkinter import ttk

def search_films():
    keyword = entry.get()
    params = {"keyword": keyword}
    response = requests.get(url, params=params, headers=headers)
    films_data = response.json()
    for widget in root.winfo_children():
        widget.destroy()

    if 'films' in films_data and isinstance(films_data['films'], list):
        descriptions = [film.get('description', '') for film in films_data['films']]
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(descriptions)
        
        num_clusters = 5  
        kmeans = KMeans(n_clusters=num_clusters, n_init=10)
        kmeans.fit(X)

        clusters = kmeans.labels_
        cluster_films = {i: [] for i in range(num_clusters)}
        
        for i, film in enumerate(films_data['films']):
            if 'nameRu' in film and 'posterUrl' in film and 'posterUrlPreview' in film:
                cluster_films[clusters[i]].append(film)

        
        for cluster_num, films_in_cluster in cluster_films.items():
            frame = ttk.Frame(root, padding="5")
            frame.grid(row=cluster_num // 4, column=cluster_num % 4, sticky="nsew")

            for film in films_in_cluster:
                poster_url = film['posterUrlPreview']
                response = requests.get(poster_url)
                img_data = response.content
                img = Image.open(BytesIO(img_data))
                img = img.resize((100, 150), Image.BILINEAR)
                photo = ImageTk.PhotoImage(img)
        
                label = ttk.Label(frame, image=photo, text=film['nameRu'], compound='top')
                label.image = photo
                label.bind('<Button-1>', lambda e, url=film['posterUrl']: open_poster(url))
                label.pack()


    else:
        output.delete(1.0, tk.END)
        output.insert(tk.END, "Некорректные данные о фильмах от API")

def open_poster(url):
    response = requests.get(url)
    img_data = response.content
    img = Image.open(BytesIO(img_data))
    img.show()

root = tk.Tk()
root.title("Dear B")

label = tk.Label(root, text="Кілт сөзді енгізіңіз:")
label.pack()

entry = tk.Entry(root)
entry.pack()

search_button = tk.Button(root, text="Іздеу", command=search_films)
search_button.pack()

output = tk.Text(root, height=200, width=500)
output.pack()

api_key = "71c5dd47-2ab2-40d4-bb00-4974097af5b6"
url = "https://kinopoiskapiunofficial.tech/api/v2.1/films/search-by-keyword"
headers = {"X-API-KEY": api_key}

root.mainloop()
