# Nirmaan-Case-Study
Introduction:
The task which was given was to create a tool to assess a student’s spoken introduction based on the transcript given. I had to follow the given rubric, generate an overall score and as the focus was on product thinking I thought to design a easy to use, modular and a model which does correct evaluation.

Thought Process:
I divided the problem into several layers like, firstly I understood the given rubric and extracted measurable features, designed a clean, visually appealing and aesthetic Interface which also produce accurate results.

Rubric Interpretation:
The rubric has five main categories:

Criterion	            Weight

Content & Structure	   30
Communication Skills	 25
Language & Grammar	   20
Confidence	           15
Pronunciation	         10

My Technical Solution Design:
I created the evaluator as a hybrid rule based and semantic NLP scoring engine.
This ensures reliability even without complex machine learning models.

Input Processing:
Accept the transcript through a text box or .txt upload
Clean up whitespace and normalize the text
Tokenize using regex to handle contractions like "I'm" and "don't." This prepares the transcript for consistent feature extraction.

Feature Extraction:
I included the following linguistic features:
Word Count: It is used to determine if the introduction has enough depth.
Type Token Ratio (TTR): TTR = unique words / total words
This helps to estimate the level of vocabulary usage
Filler Word Detection: Searches for words like um, uh, like, actually, etc…
This has a direct impact on Confidence and an indirect impact on Fluency of the speaker.
Punctuation Based Confidence:
1.Exclamation marks
2.Sentence endings
3.Storytelling rhythm
This gives indications about the speaker's confidence.

Scoring Engine:
Each rubric item has its own dedicated scoring formula.
Content and Structure based on length score.
Communication Skills: Based on clarity and lack of abrupt sentence fragments.
Language and Grammar
Formula used:
Grammar Score = 0.7 * (TTR * 100) + 0.3 * (1 – filler rate) * 100
Confidence: finds excessive filler words and rewards fluid writing.
Pronunciation is evaluated from the text flow, as audio is not available.

User Interface:
I developed a clean and modern Streamlit UI with:
Centered layout
Shadowed card blocks
Rubric table display
Scoring details and JSON section
I designed the UI to be simple and friendly, allowing anyone to use it without needing technical skills.

Output:
The tool generates a Final Score and JSON format output.

Declaration: I declare that all the work in this case study has been completed by me. I have worked on similar projects before, giving me the understanding and experience needed to carry out this work effectively and independently. 

