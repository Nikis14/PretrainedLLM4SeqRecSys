PROMPTS_DICT = {
    "long": {
        "kion": {
            "update_profile": """
    You are a smart analyst that can infer user interests from sequence of their interactions with movies. You will be presented with current user profile describing user interests and sequence of new movies watched, containing their metadata including percent of the total movie watched. Your job is to provide a general summarization of users preferences from both current user profile and new interactions.

The response should be split into 5 paragraphs following the structure of the given profile: 
1. The analysis of movie preferences based on its release year and country.
2. Genres, keywords and plot twists.
3. Characterise completed movies (watching percent more than 80%). What do these completed movies have in common?
4. Describe the movies with user "didn't finish" (watching percent less than 50%). What do these not-finished movies have in common?
5. Give short summary of user interests covering all aspects.

If there are some controversial information between current user profile and new interactions, try to resolve it.
Better include info about new preferences, since they can change over time.

Generated user's profile as a general description of the user's preferences should avoid mentioning the actual movies and ratings or user personal data. Try to predict favourite users actor, director,
genre, etc. The user profile should be useful for further movie recommendations and it should be precisely 6 paragraphs long.

The profile should include characteristics of the most relevant types of movies to the user. Do NOT mention specific movies in response.

The answer should contain exactly 5 clear and concise paragraphs.

**Current profile**:
{user_profile}

**History of movies watched**:
{user_history}""",
            # ------------------
            "create_first_profile": """
    You are a smart analyst that can infer user interests from sequence of their interactions with movies. You will be presented with sequence of movies watched and their metadata including percent of the total movie watched. Your job is to provide a general summarization of users preferences.

The response should be split into 5 paragraphs: 
1. The analysis of movie preferences based on its release year and country.
2. Genres, keywords and plot twists.
3. Characterise completed movies (watching percent more than 80%). What do these completed movies have in common?
4. Describe the movies with user "didn't finish" (watching percent less than 50%). What do these not-finished movies have in common?
5. Give short summary of user interests covering all aspects.

Generated user's profile as a general description of the user's preferences should avoid mentioning the actual movies and ratings or user personal data. Try to predict favourite users actor, director, genre, etc. The user profile should be useful for further movie recommendations and it should be precisely 6 paragraphs long.

The profile should include characteristics of the most relevant types of movies to the user. Do NOT mention specific movies in response.

The answer should contain exactly 5 clear and concise paragraphs.

**History of movies watched**:
{user_history}""",
        },
        "beauty": {
            "update_profile": """\
You are a smart analyst that can infer user interests from a sequence of their interactions with beauty products. You will be presented with the current user profile describing user preferences and a sequence of new product interactions, containing their metadata including user ratings. Your job is to provide a general summarization of the user's preferences based on both the current profile and new interactions.

The response should be split into 5 paragraphs following the structure of the given profile:
1. Analysis of product preferences based on categories and brands.
2. Types of products favored, including specific features or ingredients.
3. Characterize highly rated products (rating more than 4). What do these highly rated products have in common?
4. Describe products with low ratings (rating less than 2). What do these low-rated products have in common?
5. Provide a short summary of user interests covering all aspects.

If there is any conflicting information between the current user profile and new interactions, try to resolve it.
Emphasize new preferences, as they can evolve over time.

The generated user profile should be a general description of the user's preferences, avoiding mention of actual products, ratings, or personal data. Try to predict favorite brands, product types, ingredients, etc. The user profile should be useful for further product recommendations and should be precisely 5 paragraphs long.

The profile should include characteristics of the most relevant types of beauty products for the user. Do NOT mention specific products in response.

The answer should contain exactly 5 clear and concise paragraphs.

**Current profile**:
{user_profile}

**History of product interactions**:
{user_history}
""",
            "create_first_profile": """\
You are a smart analyst that can infer user interests from a sequence of their interactions with beauty products. You will be presented with a sequence of products interacted with and their metadata including user ratings. Your job is to provide a general summarization of the user's preferences.

The response should be split into 5 paragraphs:
1. Analysis of product preferences based on categories and brands.
2. Types of products favored, including specific features or ingredients.
3. Characterize highly rated products (rating more than 4). What do these highly rated products have in common?
4. Describe products with low ratings (rating less than 2). What do these low-rated products have in common?
5. Provide a short summary of user interests covering all aspects.

The generated user profile should be a general description of the user's preferences, avoiding mention of actual products, ratings, or personal data. Try to predict favorite brands, product types, ingredients, etc. The user profile should be useful for further product recommendations and should be precisely 5 paragraphs long.

The profile should include characteristics of the most relevant types of beauty products for the user. Do NOT mention specific products in response.

The answer should contain exactly 5 clear and concise paragraphs.

**History of product interactions**:
{user_history}
""",
        },
        "rees46": {
            "update_profile": """
You are a smart analyst that can infer user interests from a sequence of their interactions with an online store. You will be presented with the current user profile describing user interests and a sequence of new interactions, containing metadata about products including category code, brand, and price.

Your job is to provide a general summarization of user preferences from both the current user profile and new interactions.

The response should be split into 5 paragraphs following the structure of the given profile:
1. Product categories and subcategories that the user is most interested in.
2. Analysis of user preferences based on product brands.
3. Characterize products with high purchase frequency (more than 2 purchases). What do these products have in common?
4. Describe products that the user has shown little interest in (less than 2 interactions). What do these products have in common?
5. Give a short summary of user interests covering all aspects.

If there are some controversial information between the current user profile and new interactions, try to resolve it. Better include info about new preferences, since they can change over time.

Generated user's profile as a general description of the user's preferences should avoid mentioning the actual products or user personal data. Try to predict the user's favorite product categories, brands, and price ranges. The user profile should be useful for further product recommendations and should be precisely 6 paragraphs long.

The profile should include characteristics of the most relevant types of products to the user. Do NOT mention specific products in response.

The answer should contain exactly 5 clear and concise paragraphs.

**Current profile**:
{user_profile}

**History of interactions**:
{user_history}""",
            "create_first_profile": """
You are a smart analyst that can infer user interests from a sequence of their interactions with an online store. You will be presented with a sequence of products interacted with and their metadata including category code, brand, and price.

Your job is to provide a general summarization of user preferences.

The response should be split into 5 paragraphs:
1. Product categories and subcategories that the user is most interested in.
2. Analysis of user preferences based on product brands.
3. Characterize products with high purchase frequency (more than 2 purchases). What do these products have in common?
4. Describe products that the user has shown little interest in (less than 2 interactions). What do these products have in common?
5. Give a short summary of user interests covering all aspects.

Generated user's profile as a general description of the user's preferences should avoid mentioning the actual products or user personal data. Try to predict the user's favorite product categories, brands, and price ranges. The user profile should be useful for further product recommendations and should be precisely 6 paragraphs long.

The profile should include characteristics of the most relevant types of products to the user. Do NOT mention specific products in response.

The answer should contain exactly 5 clear and concise paragraphs.

**History of interactions**:
{user_history}""",
        },
    },
    "short": {},
}
