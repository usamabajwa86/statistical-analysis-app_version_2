import plotly.express as px

def heatmap(df):
    corr = df.corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto")
    return fig

def scatter_plot(df, x, y):
    fig = px.scatter(df, x=x, y=y, trendline="ols")
    return fig

def pca_scatter(df, pc1, pc2):
    fig = px.scatter(df, x=pc1, y=pc2)
    return fig
