PROMPTS_DICT = {
    "long": {
        "kion": {
            "create_profile": """\
You are a smart analyst that can infer user interests from sequence of their interactions with movies. You will be presented with sequence of movies watched and their metadata including percent of the total movie watched. Your job is to provide a general summarization of users preferences.

The response should be split into 5 paragraphs: 
1. The analysis of movie preferences based on its release year and country.
2. Genres, keywords and plot twists.
3. Characterise completed movies (watching percent more than 80%) if there are any. What do these completed movies have in common?
4. Describe the movies with user "didn't finish" (watching percent less than 50%). What do these not-finished movies have in common?
5. Give short summary of user interests covering all aspects.

Generated user's profile as a general description of the user's preferences should avoid mentioning the actual movies and ratings or user personal data. Try to predict favourite users actor, director, genre, etc. The user profile should be useful for further movie recommendations and it should be precisely 6 paragraphs long.

The profile should include characteristics of the most relevant types of movies to the user. Do NOT mention specific movies in response.

The answer should contain exactly 5 clear and concise paragraphs.

**History of movies watched**:
{user_history}""",
            # ------------------
            "aggregate_profiles": """
    You are a smart analyst that can infer user interests based on provided information about them. You will be presented with a set of user profiles, that describe user's preferences in sequential periods of time. 
    Your job is to write a general non-controversial profile aggregating information from given profiles. 

The response should be split into 5 paragraphs in the same way how it is organized in the given profiles: 
1. The analysis of movie preferences based on its release year and country.
2. Genres, keywords and plot twists.
3. Characterise completed movies (watching percent more than 80%). What do these completed movies have in common?
4. Describe the movies with user "didn't finish" (watching percent less than 50%). What do these not-finished movies have in common?
5. Give short summary of user interests covering all aspects.

If there are some controversial information, try to resolve it.

Generated user's profile as a general description of the user's preferences should avoid mentioning the actual movies and ratings or user personal data. Try to predict favourite users actor, director, genre, etc. The user profile should be useful for further movie recommendations and it should be precisely 6 paragraphs long.

The profile should include characteristics of the most relevant types of movies to the user. Do NOT mention specific movies in response.

The answer should contain exactly 5 clear and concise paragraphs.

**List of profiles for the user based on sequential time intervals**:
{user_profiles}
""",
        },
        "beauty": {
            "create_profile": """\
You are a smart analyst that can infer user interests from the sequence of their interactions with beauty products. You will be presented with a list of beauty product reviews and ratings, including product titles, descriptions, and categories. Your job is to provide a general summarization of the user's preferences based on these interactions.

The response should be split into 5 paragraphs:
1. Analyze the user's preferences based on product categories and types (e.g., skincare, makeup, haircare). What trends do you observe in the types of products reviewed?
2. Highlight the features and attributes of products that the user tends to favor, such as ingredients, brand names, or specific benefits mentioned in the reviews.
3. Characterize products that received high ratings (4 stars and above). What do these highly-rated products have in common in terms of the user's preferences?
4. Describe products that received low ratings (2 stars and below). What commonalities do these products share that might explain the user's dissatisfaction?
5. Provide a short summary of the user's interests, covering all aspects of their preferences in beauty products.

The profile should avoid mentioning specific products or ratings and should predict the user's favorite product categories, brands, and features. The user profile should be useful for further recommendations and must consist of exactly 5 clear and concise paragraphs.

**History of beauty product interactions**:
{user_history}
""",
            # ------------------
            "aggregate_profiles": """\
You are a smart analyst that can infer user interests based on provided information about them. You will be presented with a set of user profiles, each describing the user's preferences in beauty products over time. Your job is to write a general, non-controversial profile aggregating information from these profiles.

The response should be split into 5 paragraphs in the same way how it is organized in the given profiles:
1. Analyze the user's preferences based on product categories and types. Are there any noticeable trends in the categories favored over time?
2. Highlight common features and attributes that the user has consistently preferred in beauty products, such as specific ingredients or product benefits.
3. Characterize products that received high ratings (4 stars and above) across the profiles. What do these products share in terms of user appeal?
4. Describe products that received low ratings (2 stars and below) across the profiles. What patterns or characteristics do these products exhibit that might explain the user's low ratings?
5. Provide a comprehensive summary of the user's interests in beauty products, synthesizing insights from all the profiles.

The profile should avoid mentioning specific products or ratings and should predict the user's favorite categories, brands, and features. The aggregated profile should be useful for further recommendations and must consist of exactly 5 clear and concise paragraphs.

**List of profiles for the user based on sequential time intervals**:
{user_profiles}
""",
        },
        "rees46": {
            "create_profile": """\
You are a smart analyst that can infer user shopping preferences from a sequence of their interactions with an online store. You will be presented with a sequence of items viewed or purchased and their metadata, including category, brand, and price. Your job is to provide a general summarization of the user's shopping behavior.

The response should be split into 5 paragraphs:
1. Analyze the user's preferred product categories based on the frequency and variety of items viewed or purchased.
2. Describe the user's brand preferences, including whether they show loyalty to specific brands or if they tend to explore multiple brands.
3. Analyze the user's price range preferences. Do they tend to buy more expensive items, or do they focus on budget-friendly options? Are there any noticeable trends in their spending?
4. If applicable, describe any patterns in the types of items that the user frequently abandons or does not complete a purchase for. What do these items have in common (e.g., high price, specific categories, certain brands)?
5. Give a short summary of the user's overall shopping habits and preferences, including potential motivations for their behavior (e.g., brand loyalty, budget-consciousness, product variety exploration).

The generated profile should avoid mentioning specific products or order details. The profile should be useful for future shopping recommendations and retail analysis. The answer should be exactly 5 paragraphs long.

**History of items viewed or purchased**:
{user_history}
""",
            # ------------------
            "aggregate_profiles": """\
You are a smart analyst that can infer user shopping preferences based on provided information about them. You will be presented with a set of user profiles that describe the user's preferences over different time periods. Your job is to write a general, non-controversial profile by aggregating information from the given profiles.

The response should be split into 5 paragraphs in the same way that it is organized in the given profiles:
1. Analyze the user's preferred product categories based on the frequency and variety of items they interact with.
2. Describe the user's brand preferences, identifying any shifts or consistent patterns of brand loyalty or exploration.
3. Analyze the user's price range preferences. Are there any changes or consistent trends in their spending behavior across the profiles?
4. If applicable, describe patterns in the types of items that the user abandoned or did not complete a purchase for over time. What do these items have in common?
5. Provide a short summary of the user's overall shopping habits and preferences, including potential shifts in behavior over time and any suggestions for future recommendations.

If there is conflicting information across profiles, try to resolve it or highlight possible explanations for the differences.

The generated profile should avoid mentioning specific products or order details. The profile should be useful for future shopping recommendations and retail analysis. The answer should be exactly 5 paragraphs long.

**List of profiles for the user based on sequential time intervals**:
{user_profiles}
""",
        },
        "ml-20m": {
            "create_profile": """\
You are a smart analyst that can infer user movie preferences from their viewing and rating history. You will be presented with a list of movies the user has interacted with, including their titles, release years, genres, and ratings. Your task is to provide a general summary of the user's preferences based on these interactions.

The response should be split into 5 paragraphs:
1. Analyze the user's preferences based on movie genres (e.g., action, comedy, drama). Which genres does the user seem to prefer, and are there any noticeable patterns in their genre interests?
2. Highlight the user's preferences for movies based on release years. Do they favor older classics, recent releases, or movies from a particular era?
3. Characterize movies that received high ratings (4 stars and above). What patterns can you observe among these highly-rated movies in terms of genres, release years, or other possible trends?
4. Describe movies that received low ratings (2 stars and below). Are there any common characteristics (e.g., genres, release years) among these movies that might explain the user's dissatisfaction?
5. Provide a concise summary of the user's overall movie preferences. Cover their favorite genres, time periods, and the characteristics of movies they tend to enjoy the most.

The profile should avoid mentioning specific movies or ratings and should predict the user's favorite genres, release year ranges, and movie characteristics. The user profile should be useful for further movie recommendations and must consist of exactly 5 clear and concise paragraphs.

**History of movie interactions**:
{user_history}
""",
            # ------------------
            "aggregate_profiles": """\
You are a smart analyst that can infer user preferences based on provided profiles of their movie-watching history. You will be presented with a set of user profiles, each summarizing the user's preferences over different time periods. Your task is to write a general, non-controversial profile aggregating information from these profiles.

The response should be split into 5 paragraphs in the same way as the given profiles:
1. Analyze the user's preferences based on movie genres. Are there any consistent trends or changes in genre preferences over time?
2. Highlight patterns in the user's preferences for movies based on release years. Do they consistently favor movies from certain time periods, or have their preferences shifted over time?
3. Characterize movies that received high ratings (4 stars and above) across the profiles. What do these movies have in common in terms of genres, release years, or other notable features?
4. Describe movies that received low ratings (2 stars and below) across the profiles. Are there any recurring patterns or trends that might explain the user's low ratings?
5. Provide a comprehensive summary of the user's overall movie preferences, synthesizing insights from all the profiles. Cover their favored genres, preferred release year ranges, and any other notable trends.

The profile should avoid mentioning specific movies or ratings and should predict the user's favorite genres, time periods, and movie characteristics. The aggregated profile should be useful for further movie recommendations and must consist of exactly 5 clear and concise paragraphs.

**List of profiles for the user based on sequential time intervals**:
{user_profiles}
""",},
   "amazon-m2": {
         "create_profile": """\
You are a smart analyst that infers user preferences from their shopping interactions. Analyze the user's behavior using their item history, focusing on metadata: **Title, Price, Brand, Color, Size, Model, Material, and Item Description**. Generate a profile split into 5 paragraphs:

1. **Product Preferences**: Identify preferred product types (e.g., electronics, apparel) inferred from recurring keywords in Titles/Descriptions, Material usage, and Size/Color patterns. Highlight frequency vs. diversity.
2. **Brand Affinity**: Analyze loyalty to specific Brands or exploration across brands. Note consistency in Brand-Color or Brand-Material combinations.
3. **Price Sensitivity**: Determine preferred price ranges (budget, mid-tier, luxury) and spending trends. Compare prices of purchased vs. abandoned items.
4. **Abandonment Patterns**: Identify attributes (e.g., high Price, unpopular Color/Size, niche Materials) of frequently abandoned items. Contrast with purchased items.
5. **Behavior Summary**: Synthesize motivations (e.g., practicality vs. luxury, brand loyalty vs. experimentation) and suggest recommendation strategies.

Avoid mentioning specific products. Keep the profile concise and actionable. Response must be **exactly 5 paragraphs**.

**User History**:
{user_history}
""",
    # ------------------
    "aggregate_profiles": """\
You are an analyst tasked with unifying sequential user profiles. Aggregate insights from the profiles below into one coherent summary, split into 5 paragraphs:

1. **Category Evolution**: Highlight trends in product type preferences over time (e.g., shifts from apparel to electronics). Resolve conflicts (e.g., "user prefers both casual and formal wear").
2. **Brand Dynamics**: Track Brand loyalty changes (e.g., from BudgetBrand to PremiumBrand) or new explorations. Note seasonal or price-linked patterns.
3. **Price Adaptations**: Identify spending habit shifts (e.g., increased luxury purchases) or consistent budget constraints. Explain contradictions.
4. **Abandonment Trends**: Compare abandonment patterns across periods (e.g., initially abandoning high-priced items, later abandoning wrong Sizes). Find common threads.
5. **Holistic Profile**: Summarize overarching behavior (e.g., "transitioning from budget-conscious to quality-driven") and recommend personalized strategies.

Resolve conflicts logically (e.g., "higher spending in Q4 suggests holiday splurging"). Response must be **exactly 5 paragraphs**.

**User Profiles**:
{user_profiles}
""",
}

    },
    "short": {
        "kion": {
            "create_profile": """\
You are an advanced movie recommender system designed to generate structured user profiles based on movie interactions. Your task is to summarize a user's interaction history with movies into a concise, structured profile. Each movie has the following attributes: Name, Release Year, Genres, Countries, Keywords, and Director. Follow the format below:

### User Profile:
1. **Preferences**:
   - **Genres**: List the top genres with approximate percentages of preference.
   - **Release Years**: Group movies by release year ranges and include approximate percentages.
   - **Countries**: Indicate the user’s preferred countries of movie production and their approximate proportions.
   - **Keywords**: Identify recurring themes or keywords from the user’s liked movies.
   - **Directors**: List the most-watched directors and the number of movies associated with each.
2. **Behavioral Insights**: Highlight notable trends in the user’s behavior, such as rewatching, exploration of niche categories, or franchise preferences.
3. **Engagement Trends**: Describe the user’s preferences in terms of plot complexity, themes, or other qualitative attributes.

### Instructions:
- Ensure the profile is concise, well-structured, and uses bullet points for clarity.
- Use approximate percentages or counts to quantify preferences and trends.
- Base the profile on the user's interaction history and inferred preferences.
- Avoid redundancy and focus on meaningful insights about the user's preferences and behaviors.

**History of movies watched**:
{user_history}""",
            # ------------------
            "aggregate_profiles": """\
You are a smart analyst that can infer user interests based on provided information about them. You will be presented with a set of user profiles, that describe user's preferences in sequential periods of time. 
Your job is to write a general non-controversial profile aggregating information from given profiles. 

The response should be split into 3 paragraphs in the same way how it is organized in the given profiles: 
1. The analysis of movie preferences based on its release year and country.
2. Genres, keywords and plot twists.
3. Characterise completed movies (watching percent more than 80%). What do these completed movies have in common?
4. Describe the movies with user "didn't finish" (watching percent less than 50%). What do these not-finished movies have in common?
5. Give short summary of user interests covering all aspects.

If there are some controversial information, try to resolve it.

Generated user's profile as a general description of the user's preferences should avoid mentioning the actual movies and ratings or user personal data. Try to predict favourite users actor, director, genre, etc. The user profile should be useful for further movie recommendations.

The profile should include characteristics of the most relevant types of movies to the user. Do NOT mention specific movies in response.

**List of profiles for the user based on sequential time intervals**:
{user_profiles}""",
        },
        "beauty": {
            "create_profile": """\
You are an advanced beauty product recommender system designed to generate structured user profiles based on user interactions with beauty products. Your task is to summarize a user's interaction history with beauty products into a concise, structured profile. Each product has the following attributes: Title, Description, and Categories. Follow the format below:

### User Profile:
1. **Preferences**:
   - **Categories**: List the top beauty categories the user engages with (e.g., Skincare, Makeup, Haircare) along with approximate percentages of preference.
   - **Product Features**: Identify preferred product features or attributes inferred from product descriptions (e.g., moisturizing, long-lasting, hypoallergenic) with approximate proportions.
   - **Brands**: List the most-engaged brands and the number of products associated with each.
   - **Price Range**: Indicate the typical price range of products the user interacts with (e.g., Budget, Mid-Range, Premium).

2. **Behavioral Insights**:
   - Highlight notable trends in the user’s behavior, such as preference for natural ingredients, exploration of new brands, loyalty to specific brands, seasonal buying patterns, or preference for specific product formulations.

3. **Engagement Trends**:
   - Describe the user’s preferences in terms of product complexity, desired outcomes (e.g., anti-aging, hydration, volumizing), frequency of purchases, and any other qualitative attributes.

### Instructions:
- Ensure the profile is concise, well-structured, and uses bullet points for clarity.
- Use approximate percentages or counts to quantify preferences and trends.
- Base the profile on the user's interaction history and inferred preferences.
- Avoid redundancy and focus on meaningful insights about the user's preferences and behaviors.

**History of beauty products interacted with**:
{user_history}""",
            "aggregate_profiles": """\
You are a smart analyst that can infer user interests based on provided information about them. You will be presented with a set of user profiles that describe the user's preferences in sequential periods of time. 

Your job is to write a general, non-controversial profile aggregating information from the given profiles.

The response should be organized into sections in the same way how the individual profiles are structured:

1. **Preferences**:
   - **Categories**: Aggregate the top beauty categories and their approximate preferences across all profiles.
   - **Product Features**: Summarize key product features or attributes the user consistently prefers.
   - **Brands**: Combine brand preferences, highlighting the most frequent or favored brands.
   - **Price Range**: Consolidate typical price ranges from different periods to identify the user's spending patterns.

2. **Behavioral Insights**:
   - Highlight overarching behavioral trends, such as brand loyalty, ingredient preferences, purchasing patterns, exploration of new products, or seasonal buying habits.

3. **Engagement Trends**:
   - Describe consistent preferences in terms of product complexity, desired outcomes (e.g., anti-aging, hydration, volumizing), frequency of purchases, and other qualitative attributes.

4. **Summary**:
   - Provide a short summary of the user's interests covering all aspects, ensuring it encompasses the main preferences and behaviors identified.

### Instructions:
- Ensure the aggregated profile is cohesive, well-structured, and maintains clarity.
- Resolve any conflicting information by identifying the most dominant trends or preferences.
- Avoid mentioning specific products or personal data.
- Focus on predicting favorite brands, categories, product features, and other relevant aspects.
- The user profile should be useful for further beauty product recommendations.

**List of profiles for the user based on sequential time intervals**:
{user_profiles}""",
        },
        "rees46": {
            "create_profile": """\
You are an advanced recommender system designed to generate structured user profiles based on interactions with electronic products. Your task is to summarize a user's interaction history with electronic items into a concise, structured profile. Each interaction has the following attributes: Category, Brand, Price, and Event Type. Follow the format below:

### User Profile:
1. **Preferences**:
   - **Product Categories**: List the top categories with approximate percentages of preference.
   - **Brands**: Identify preferred brands and their approximate proportions.
   - **Price Ranges**: Group interactions by price ranges (e.g., <$50, $50-$200, $200-$500, >$500) with approximate percentages.
   - **Event Types**: Indicate the types of interactions (e.g., viewed, added to cart, purchased, wish-listed) and their approximate proportions.

2. **Behavioral Insights**: Highlight notable trends in the user’s behavior, such as frequent purchases, brand loyalty, price sensitivity, or preference for specific product features.

3. **Engagement Trends**: Describe the user’s engagement patterns, such as peak shopping times, repeat interactions with certain categories or brands, or responsiveness to promotions and discounts.

### Instructions:
- Ensure the profile is concise, well-structured, and uses bullet points for clarity.
- Use approximate percentages or counts to quantify preferences and trends.
- Base the profile on the user's interaction history and inferred preferences.
- Avoid redundancy and focus on meaningful insights about the user's preferences and behaviors.

**History of interactions with electronic products**:
{user_history}""",
            "aggregate_profiles": """\
You are a smart analyst tasked with inferring comprehensive user interests based on provided segmented user profiles. You will be presented with a set of user profiles, each describing the user's preferences and behaviors in sequential time intervals. 

Your job is to create a general, non-controversial profile that aggregates information from the given profiles. 

### Aggregated User Profile:
1. **Overall Preferences**:
   - **Product Categories**: Summarize the user's preferred categories with combined approximate percentages.
   - **Brands**: Consolidate preferred brands and their overall proportions.
   - **Price Ranges**: Aggregate price range preferences across all segments.
   - **Event Types**: Combine the different types of interactions and their overall proportions.

2. **Behavioral Patterns**:
   - Identify consistent behaviors such as brand loyalty, frequent purchasing habits, price sensitivity, or preference for specific product features.
   - Highlight any emerging trends or shifts in user behavior over time.

3. **Engagement Insights**:
   - Describe the user's overall engagement patterns, including peak shopping times, responsiveness to promotions, and repeat interactions with specific categories or brands.
   - Note any significant changes in engagement across different time intervals.

4. **Summary**:
   - Provide a short summary of the user's interests covering all aspects, ensuring it encompasses the main preferences and behaviors identified.

### Instructions:
- Ensure the aggregated profile is concise, well-structured, and organized into clear sections as outlined above.
- Use approximate percentages or counts to quantify aggregated preferences and behaviors.
- Resolve any conflicting information by identifying dominant trends or providing balanced insights.
- Avoid mentioning specific products, brands, or personal data unless necessary for understanding user preferences.
- The profile should be useful for further recommendations and marketing strategies.

**List of segmented user profiles based on sequential time intervals**:
{user_profiles}""",
        },
        "ml-20m": {
            "create_profile": """\
You are an advanced movie recommender system designed to generate structured user profiles based on movie interactions. Your task is to summarize a user's interaction history with movies into a concise, structured profile. Each movie has the following attributes: Name, Release Year and Genres. Follow the format below:

### User Profile:
1. **Preferences**:
   - **Genres**: List the top genres per user.
   - **Release Years**: Group movies by release year and epochs.
2. **Behavioral Insights**: Highlight notable trends in the user’s behavior, such as rewatching, exploration of niche categories, or franchise preferences.
3. **Engagement Trends**: Describe the user’s preferences in terms of plot complexity, themes, or other qualitative attributes.

### Instructions:
- Ensure the profile is concise, well-structured, and uses bullet points for clarity.
- Base the profile on the user's interaction history and inferred preferences.
- Avoid redundancy and focus on meaningful insights about the user's preferences and behaviors.

**History of movies watched**:
{user_history}""",
            # ------------------
            "aggregate_profiles": """\
You are a smart analyst that can infer user interests based on provided information about them. You will be presented with a set of user profiles, that describe user's preferences in sequential periods of time. 
Your job is to write a general non-controversial profile aggregating information from given profiles. 

The response should be split into 3 paragraphs in the same way how it is organized in the given profiles: 
1. The analysis of movie preferences based on its release year and style.
2. Genres, keywords and plot twists.
3. Give short summary of user interests covering all aspects.

If there are some controversial information, try to resolve it.

Generated user's profile as a general description of the user's preferences should avoid mentioning the actual movies and ratings or user personal data. Try to predict favourite users genre, epochs, directors, etc. The user profile should be useful for further movie recommendations.

The profile should include characteristics of the most relevant types of movies to the user. Do NOT mention specific movies in response.

**List of profiles for the user based on sequential time intervals**:
{user_profiles}""",
        },
        "amazon-m2": {
            "create_profile": """\
You are an advanced recommender system designed to generate structured user profiles based on interactions with shopping items. Your task is to summarize a user's interaction history with shopping items into a concise, structured profile. Each interaction includes the following attributes: Title, Price, Brand, Color, Size, Model, Material, and Item Description. Use this information to infer the user's preferences, behaviors, and engagement trends. Follow the format below:

### User Profile:
1. **Preferences**:
   - **Product Categories**: Deduce the most interacted-with product categories (e.g., clothing, electronics, furniture) and their approximate proportions.
   - **Brands**: Identify preferred brands with approximate proportions.
   - **Price Ranges**: Categorize interactions by price ranges (e.g., <$20, $20-$50, $50-$100, >$100) and their approximate proportions.
   - **Colors, Sizes, and Materials**: Summarize the user's favorite colors, sizes, and materials based on interactions.
   - **Features**: Highlight recurring features of interest (e.g., eco-friendly materials, specific styles, or popular models).

2. **Behavioral Insights**: Identify notable patterns in the user's behavior, such as:
   - Strong preferences for specific product types, brands, or features.
   - Price sensitivity or willingness to spend on premium items.
   - Any apparent seasonal or event-driven shopping patterns.

3. **Engagement Trends**: Describe the user's engagement patterns, such as:
   - Frequency of interactions (e.g., daily, weekly).
   - Popular times for shopping.
   - Interaction types (e.g., viewed, added to cart, purchased, saved for later).
   - Trends in exploring new products or sticking to familiar ones.

### Instructions:
- Ensure the profile is concise, well-structured, and uses bullet points for clarity.
- Use approximate percentages or counts to quantify preferences and trends.
- Base the profile on the user's interaction history and inferred preferences.
- Avoid redundancy and focus on meaningful insights about the user's preferences and behaviors.

**History of interactions with shopping items**:
{user_history}""",
            # ------------------
            "aggregate_profiles": """\
You are a smart analyst that can infer user interests based on provided information about them. You will be presented with a set of user profiles that describe a user's shopping preferences and behavior during sequential time intervals. Your task is to write a general, non-controversial profile by aggregating and summarizing the information from these profiles.

The response should be organized into the following sections:

1. **Overall Preferences**:
   - Summarize the user's preferred product categories, brands, price ranges, colors, sizes, materials, and recurring product features based on the profiles.
   - Mention any noticeable shifts in preferences or consistency over time.

2. **Behavioral Insights**:
   - Highlight general behavioral patterns, such as shopping frequency, price sensitivity, loyalty to brands, or openness to exploring new products.
   - Identify any overarching trends, such as seasonal shopping habits or preferences for specific product types (e.g., casual vs. formal wear, tech gadgets, or home decor).

3. **Engagement Trends**:
   - Summarize how the user's engagement has evolved over time (e.g., increasing purchases, shifts in interaction types like viewing vs. purchasing).
   - Identify any notable changes in shopping behavior, such as trying new brands, exploring different product features, or shifting price ranges.

4. **Concise Summary**:
   - Provide a brief, high-level summary of the user's overall shopping interests and behaviors, covering preferences, behavioral patterns, and engagement trends.

### Instructions:
- Focus on general patterns and insights rather than specific items or time periods.
- Resolve any conflicting information by identifying the most consistent trends.
- Avoid mentioning specific products, brands, or user information directly.
- Ensure the profile is useful for enhancing future product recommendations.

**List of user profiles based on sequential time intervals**:
{user_profiles}""",
        },
    },
    "user_needs": {
        "rees46": {
            "create_profile": """\
You are an advanced recommender system designed to generate structured user profiles based on their interactions with electronic products. Your task is to infer a user's needs and preferences from their interaction history and summarize them into a concise, structured profile. Each interaction has the following attributes: Category, Brand, Price, and Event Type. Follow the format below:

### User Profile:
1. **Inferred Needs**:
   - **Primary Need**: Identify the user's primary need based on their interaction history (e.g., productivity, entertainment, communication).
   - **Secondary Needs**: List any secondary or related needs that the user may have based on their interactions.

2. **Preferences**:
   - **Product Categories**: List the top categories that align with the user's needs, along with approximate percentages of preference.
   - **Brands**: Identify preferred brands within the relevant categories and their approximate proportions.
   
3. **Behavioral Insights**: Highlight notable trends in the user's behavior that relate to their inferred needs, such as frequent purchases in specific categories, brand loyalty within those categories, price sensitivity, or preference for specific product features.

4. **Recommendations**: Suggest 2-3 products or categories that align with the user's inferred needs and preferences, based on their interaction history.

### Instructions:
- Focus on inferring the user's needs based on their interaction history and structure the profile accordingly.
- Ensure the profile is concise, well-structured, and uses bullet points for clarity.
- Use approximate percentages or counts to quantify preferences and trends.
- Base the profile on the user's interaction history and inferred needs and preferences.
- Avoid redundancy and focus on meaningful insights about the user's needs, preferences, and behaviors.
- Include a brief recommendations section to suggest products or categories that match the user's needs.

**History of interactions with electronic products**:
{user_history}""",
            "aggregate_profiles": """"\
You are an advanced profile aggregator for a recommender system. Your task is to analyze multiple user profiles and combine them into a single, structured summary focusing on the top-3-5 most significant user needs. Follow this format strictly:

### Aggregated User Needs Profile:

1. Top-3-5 User Needs (number of needs depends on the user profile)(ordered by priority):
[Need 1]: 
- Primary purpose
- Associated preferences in brands
- Associated preferences for price for the products in the same category as the requested

[Need 2]: 
(same structure)

[Need 3]: 
(same structure)

...

2. Cross-Need Insights:
- Common patterns
- Notable correlations
- Price sensitivity overview

Format Guidelines:
- Keep each need description under 30 words
- Use percentages for quantifiable data
- Focus on actionable insights
- Maintain consistent structure

Input Profiles:
{user_profiles}""",
        }
    },
    "user_needs_json": {
        "rees46": {
            "create_profile": """\
You are an advanced recommender system designed to generate structured user profiles based on their interactions with electronic products. Your task is to infer a user's needs and preferences from their interaction history and summarize them into a concise, structured profile. Each interaction has the following attributes: Category, Brand, Price, and Event Type. Return the profile as a JSON object. Follow the format below:

### User Profile:
```json{
  "inferred_needs": {
     "primary_need": <Identify the user's primary need based on their interaction history (e.g., productivity, entertainment, communication).>,
     "secondary_needs": <List any secondary or related needs that the user may have based on their interactions.>
  },
  "preferences": {
     "product_categories": <List the top categories that align with the user's needs, along with approximate percentages of preference.>,
     "brands": <Identify preferred brands within the relevant categories and their approximate proportions.>
  },
  "behavioral_insights": <Highlight notable trends in the user's behavior that relate to their inferred needs, such as frequent purchases in specific categories, brand loyalty within those categories, price sensitivity, or preference for specific product features.>,
  "recommendations": <Suggest 2-3 products or categories that align with the user's inferred needs and preferences, based on their interaction history.>
}```.

### Instructions:
- Focus on inferring the user's needs based on their interaction history and structure the profile accordingly.
- Ensure the profile is concise, well-structured, and uses json format.
- Base the profile on the user's interaction history and inferred needs and preferences.
- Avoid redundancy and focus on meaningful insights about the user's needs, preferences, and behaviors.
- Include a brief recommendations section to suggest products or categories that match the user's needs.

**History of interactions with electronic products**:
{user_history}""",
            "aggregate_profiles": """"\
You are an advanced profile aggregator for a recommender system. Your task is to analyze multiple user profiles and combine them into a single, structured summary focusing on the top-3-5 most significant user needs. Return the profile as a JSON object. Follow this format strictly:

### Aggregated User Needs Profile:

```json{
  "top_needs": [
     <Need 1>: {
        "primary_purpose": <Primary purpose>,
        "associated_brands": <Associated preferences in brands>,
        "associated_price": <Associated preferences for price for the products in the same category as the requested>
     },
     <Need 2>: 
     (same structure)
     <Need 3>: 
     (same structure)
     ...
  ],
  "cross_need_insights": {
     "common_patterns": <Common patterns>,
     "notable_correlations": <Notable correlations>,
     "price_sensitivity_overview": <Price sensitivity overview>
  }
}```

Format Guidelines:
- Keep each need description under 30 words
- Use percentages for quantifiable data
- Follow the json format
- Focus on actionable insights
- Maintain consistent structure

Input Profiles:
{user_profiles}""",
        }
    },
    "short_ru": {
        "kion": {
            "create_profile": """\
Ты продвинутая система рекомендаций фильмов, разработанная для создания структурированных профилей пользователей на основе их взаимодействий с фильмами. Твоя задача — суммаризировать историю взаимодействия пользователя с фильмами в ёмкий и структурированный профиль. У каждого фильма есть следующие атрибуты: Название, Год выпуска, Жанры, Страны, Ключевые слова и Режиссёр. Следуй приведённому ниже формату:

### Профиль пользователя:
1. **Предпочтения**:
    - **Жанры: Перечисли основные жанры.
    - **Годы выпуска**: Сгруппируй фильмы по диапазонам годов выпуска.
    - **Страны**: Укажи предпочтительные для пользователя страны создания фильмов.
    - **Ключевые слова**: Определи повторяющиеся темы или ключевые слова из понравившихся пользователю фильмов.
    - **Режиссёры**: Перечисли самых просматриваемых режиссёров и количество фильмов, связанных с каждым из них.
2. **Поведенческие инсайты**: Выдели важные тенденции в поведении пользователя, такие как пересмотр фильмов, исследование нишевых категорий, предпочтения франшиз или исследование жанров.
3. **Тренды вовлечённости**: Опиши предпочтения пользователя с точки зрения сложности сюжета, тем или других качественных характеристик, которые могут показать, что привликают пользователя.

### Инструкции:
- Убедись, что профиль краткий, хорошо структурированный и использует буллет-поинты для чёткости структуры.
- Основывай профиль на истории взаимодействий пользователя и предполагаемых предпочтениях.
- Избегай избыточности и сосредоточьтесь на значимых инсайтах о предпочтениях и поведении пользователя.

**История просмотренных фильмов**:
{user_history}""",
            # ------------------
            "aggregate_profiles": """\      
Ты — умный аналитик, способный выявлять интересы пользователя на основе предоставленной о нем информации. Тебе будет представлен набор из нескольких профилей одного пользователя, описывающих его предпочтения в последовательные периоды времени.

Твоя задача — написать общий, непротиворечивый профиль, объединяющий информацию из предоставленных профилей пользователя.

Ответ должен быть организован по разделам так же, как структурированы отдельные профили:

1. **Предпочтения**:
    - **Жанры: Перечисли основные жанры.
    - **Годы выпуска**: Сгруппируй фильмы по диапазонам годов выпуска.
    - **Страны**: Укажи предпочтительные для пользователя страны создания фильмов.
    - **Ключевые слова**: Определи повторяющиеся темы или ключевые слова из понравившихся пользователю фильмов.
    - **Режиссёры**: Перечисли самых просматриваемых режиссёров и количество фильмов, связанных с каждым из них.
2. **Поведенческие инсайты**: Выдели важные тенденции в поведении пользователя, такие как пересмотр фильмов, исследование нишевых категорий, предпочтения франшиз или исследование жанров.
3. **Тренды вовлечённости**: Опиши предпочтения пользователя с точки зрения сложности сюжета, тем или других качественных характеристик, которые могут показать, что привликают пользователя.

                
### Инструкции:
- Убедись, что объединенный профиль является целостным, хорошо структурированным и сохраняет ясность.
- Согласуй любую противоречивую информацию, выявив наиболее доминирующие тенденции или предпочтения.
- Избегай упоминания конкретных фильмов или личных данных.
- Сосредоточься на прогнозировании любимых фильмов, режиссеров, жанров и других соответствующих аспектов.
- Профиль пользователя должен быть полезен для дальнейших рекомендаций фильмов.

**Список профилей пользователя, основанных на последовательных временных интервалах**:
{user_profiles}""",
        }
    },
}
