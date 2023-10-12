# Information Retrieval Project
## Project Description
In this project, we will focus on the application of Information Retrieval techniques to the speeches of the Hellenic Parliament. The data has been gathered from the official website of the Hellenic Parliament, as the speeches are publicly available. The dataset can be accessed from here [Greek Parliament Proceedings](https://www.hellenicparliament.gr/Praktika/Synedriaseis-Olomeleias).

The dataset includes 1,280,918 speeches from Greek Parliament members, with a total volume of 2.30 GB, extracted from 5,355 session recording files. The temporal range spans from the beginning of July 1989 to the end of July 2020. The dataset is provided in a UTF-8 encoded CSV file, containing various columns related to members, parties, speech dates, etc.

## Objectives
Our goal is to organize and process the data to enable the extraction of useful information from these speeches. Specifically, the focus will be on the following tasks:

- **Develop a web-based application** that allows users to search for information within the speech data. The application should provide search capabilities similar to a search engine for this specific dataset.
- **Identify the most significant keywords** per speech, per parliament member, and per political party. Additionally, analyze how these keywords evolve over time.
- **Explore similarities between parliament members** by extracting feature vectors for each member and then calculating similarities between pairs to identify top-k pairs with the highest similarity (where k is a parameter).
- **Utilize Latent Semantic Indexing (LSI)** across all speeches to discover significant thematic areas. Express each speech as a vector in a multidimensional space.
- **Apply clustering** to group speeches, forming clusters with high similarity. Investigate if speeches within the same cluster share substantial commonalities.

# Running the App
- Run the Flask application using the following command:
```
python app.py
```
- The app should start, and you'll see a message indicating that the server is running.
- Open your web browser and go to `http://localhost:5000/` to access the app.