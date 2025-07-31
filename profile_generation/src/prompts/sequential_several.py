PROMPTS_DICT = {
    "short": {
        "ml-20m": {
            "user_needs": """
    ### User Profile
    # **Preferences**
    # - **Needs**: Describe the user's specific needs or preferences.
    # ### Behavioral Insights
    # - **Rewatching**: Identify movies that the user has watched multiple times.
    # - **Exploration**: Highlight notable trends in the user’s behavior, such as rewatching, exploration of niche categories, or franchise preferences.
    # - **Engagement Trends**: Describe the user’s preferences in terms of plot complexity, themes, or other qualitative attributes.
    # 
    # ### User history:
    # {user_history}""",
            "user_genres": """
    ### User Profile
    # **Preferences**
    # - **Genres**: List the top genres per user.
    # 
    # ### User history:
    # {user_history}""",
            "user_brands": """
    ### User Profile
    # **Preferences**
    # - **Brands**: Identify preferred brands within the relevant categories and their approximate proportions.
    # 
    # ### User history:
    # {user_history}""",
            "aggregate": """You are a smart analyst that can infer user interests based on provided information about them. You will be presented with a set of user profiles, that describe user's preferences in sequential periods of time. 
Your job is to write a general non-controversial profile aggregating information from given profiles. 
If there are some controversial information, try to resolve it.

Generated user's profile as a general description of the user's preferences should avoid mentioning the actual movies and ratings or user personal data. 
The profile should include characteristics of the most relevant types of movies to the user. Do NOT mention specific movies in response.

**List of profiles for the user based on sequential time intervals**:
{user_profiles}""",
        },
        "beauty": {
            "product_needs": """\
You are a dermatologist's assistant tasked with understanding a patient's skincare and beauty needs based on their product history. You will be presented with a list of beauty product reviews and ratings, including product titles, descriptions, and categories. Your job is to infer the user's primary skincare concerns and wellness goals.

The response should be split into 3 paragraphs:

1.  **Skincare Concerns:** Based on the products reviewed, identify the user's main skincare concerns. Are they focused on anti-aging, acne treatment, hydration, sensitive skin, or other specific issues? Explain your reasoning based on product categories and descriptions.
2.  **Ingredient Preferences:** Analyze the ingredients mentioned in the products the user interacts with. Do they gravitate towards natural ingredients, specific active ingredients (like retinol or hyaluronic acid), or avoid certain ingredients (like parabens or sulfates)?
3.  **Wellness & Self-Care:**  Does the user's history suggest a focus on overall wellness and self-care beyond just skincare? Do they purchase products related to relaxation, aromatherapy, or internal beauty (e.g., supplements)? Provide a concise summary of their holistic approach to beauty and wellness.

The profile should focus on the user's underlying needs and motivations related to skincare and wellness, avoiding direct mention of product names or specific ratings. This profile will be used to recommend products that address their core concerns and promote their overall well-being.

**History of beauty product interactions**:
{user_history}""",
            "product_patterns": """\
You are a personal care analyst examining a user's beauty product usage patterns. You will be presented with a list of beauty product reviews and ratings, including product titles, descriptions, and categories. Your goal is to understand their haircare routine and general beauty habits.

The response should be split into 3 paragraphs:

1.  **Haircare Routine:** Based on the products reviewed, describe the user's likely haircare routine. Do they focus on daily styling, occasional treatments, or specific hair concerns (e.g., color-treated hair, damage repair, volume)? Explain your reasoning.
2.  **Beauty Tool Preferences:**  Analyze any beauty tools or accessories in their history (e.g., brushes, hair dryers, curlers). What types of tools do they use, and what does this suggest about their beauty habits and level of expertise?
3.  **Frequency and Experimentation:** Based on the diversity and quantity of products reviewed, assess the user's frequency of product use and their willingness to experiment with new products. Are they a creature of habit or an adventurous beauty explorer?

The profile should focus on the user's practical approach to beauty, their routines, and their habits, without mentioning specific product names or ratings. This profile will help recommend products that fit seamlessly into their existing routine and cater to their level of engagement with beauty products.

**History of beauty product interactions**:
{user_history}""",
            "product_brand": """\
You are a beauty brand and style curator, skilled at discerning the user's preferred brands, aesthetic, and price point from their product interactions. You will be given a history of beauty product reviews and ratings. Your task is to analyze this history and generate a profile that emphasizes the user's brand and stylistic preferences.

The profile should consist of 4 paragraphs:

1. **Brand Engagement:** Which brands does the user interact with most frequently? Are there any emerging patterns in terms of brand preference and loyalty?
2. **Price Point Sensitivity:** Based on the products they review, what is the user's typical price point preference? Do they tend to gravitate towards luxury, mid-range, or budget brands?
3. **Style and Aesthetic:** Infer the user's preferred aesthetic or style based on product interactions. Are they attracted to minimalistic, natural, glam, or experimental looks?
4. **Brand and Style Summary:** Provide a final summary of the user's brand and aesthetic preferences, including an inference of their general price sensitivity. What kind of products align with their stylistic choices and brand preferences?

The profile must avoid specific product mentions and ratings, focusing on general brand trends and stylistic impressions. The user profile should be useful for further recommendations based on brand preferences.

**History of beauty product interactions**:
{user_history}""",
            "aggregate": """\
You are a smart analyst that can infer user interests based on provided information about them. You will be presented with a set of user profiles that describe the user's preferences in sequential periods of time. 

Your job is to write a general, non-controversial profile aggregating information from the given profiles.

The response should be organized into sections in the same way how the individual profiles are structure.

**List of profiles for the user based on sequential time intervals**:
{user_profiles}""",
        },
    }
}
