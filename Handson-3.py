import pandas as pd
import plotly.express as px

# Sample Data: Ensure df contains both sentiment and emotion predictions
df = pd.DataFrame({
    "predicted_sentiment": ["Positive", "Negative", "Positive", "Neutral", "Negative", "Positive"],
    "predicted_emotion": ["Happy", "Sad", "Surprise", "Neutral", "Angry", "Happy"]
})

# Create interactive scatter plot
fig = px.scatter(df,
                 x="predicted_sentiment",
                 y="predicted_emotion",
                 color="predicted_emotion",
                 title="Interactive Sentiment vs. Facial Emotion Analysis",
                 labels={"predicted_emotion": "Facial Emotion"},
                 hover_data=["predicted_sentiment"])  # Adds hover information

fig.show()
