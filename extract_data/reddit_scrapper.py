import os
import time
from datetime import datetime
import pandas as pd
import praw
from dotenv import load_dotenv

# --- Cargar credenciales desde .env ---
load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    username=os.getenv("REDDIT_USERNAME"),
    password=os.getenv("REDDIT_PASSWORD"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

# --- Carpeta outputs ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def download_posts(subreddit, authors, delay=3):
    """
    Descarga hasta 1000 posts más recientes de cada autor en un subreddit.

    :param subreddit: str, nombre del subreddit
    :param authors: list de str, lista de autores a consultar
    :param delay: int, segundos a esperar entre requests
    """
    for author_name in authors:
        print(f"⬇️ Descargando posts de {author_name} en r/{subreddit}...")

        posts_data = []
        author = reddit.redditor(author_name)

        try:
            for post in author.submissions.new(limit=None):
                if post.subreddit.display_name.lower() == subreddit.lower():
                    created = post.created_utc
                    posts_data.append({
                        "id": post.id,
                        "author": str(post.author),
                        "title": post.title,
                        "selftext": post.selftext,
                        "num_comments": post.num_comments,
                        "score": post.score,
                        "created_utc": datetime.fromtimestamp(created).strftime("%Y-%m-%d %H:%M:%S")
                    })

            # Guardar CSV
            filename = os.path.join(OUTPUT_DIR, f"{subreddit}_{author_name}.csv")
            if posts_data:
                pd.DataFrame(posts_data).to_csv(filename, index=False, encoding="utf-8")
                print(f"   Guardados {len(posts_data)} posts en {filename}")
            else:
                print("   No se encontraron posts")

        except Exception as e:
            print(f"   ⚠️ Error al descargar posts de {author_name}: {e}")

        # Espera entre autores
        print(f"   Esperando {delay} segundos antes del siguiente autor...\n")
        time.sleep(delay)

    print("✅ Descarga completa para todos los autores.")


# --- EJEMPLO DE USO ---
if __name__ == "__main__":
    subreddit_name = "argentina"
    authors_list = ["autor1", "autor2", "autor3"]  # reemplazá por tus autores

    download_posts(subreddit_name, authors_list, delay=3)
