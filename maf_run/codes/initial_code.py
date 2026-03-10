import pandas as pd
from plotnine import ggplot, aes, geom_bar, labs, theme, element_text, scale_fill_manual

# Data preparation
data = {
    "Number of Bits": [8, 16, 32, 64, 128] * 4,
    "Precision (%)": [
        0.7113, 0.7624, 0.7993, 0.7812, 0.7559,  # NASH
        0.747, 0.8013, 0.8418, 0.8297, 0.7924,  # NASH-DN
        0.6859, 0.7165, 0.7753, 0.7456, 0.7318,  # VDSH
        0.6616, 0.7351, 0.7554, 0.735, 0.6986   # STH
    ],
    "Model": [
        "NASH", "NASH", "NASH", "NASH", "NASH",
        "NASH-DN", "NASH-DN", "NASH-DN", "NASH-DN", "NASH-DN",
        "VDSH", "VDSH", "VDSH", "VDSH", "VDSH",
        "STH", "STH", "STH", "STH", "STH"
    ]
}

df = pd.DataFrame(data)

# Define color scheme for models
color_scheme = {
    "NASH": "#1f77b4",
    "NASH-DN": "#ff7f0e",
    "VDSH": "#2ca02c",
    "STH": "#d62728"
}

# Create the bar chart
plot = (
    ggplot(df, aes(x="Number of Bits", y="Precision (%)", fill="Model"))
    + geom_bar(stat="identity", position="dodge")
    + labs(
        title="Precision of the top 100 retrieved documents on Reuters dataset (Supervised hashing), compared with other supervised baselines.",
        x="Number of Bits",
        y="Precision (%)"
    )
    + theme(
        axis_text_x=element_text(size=10, angle=45, hjust=1),
        axis_text_y=element_text(size=10),
        axis_title_x=element_text(size=12),
        axis_title_y=element_text(size=12),
        plot_title=element_text(size=14, weight="bold", ha="center")
    )
    + scale_fill_manual(values=color_scheme)
)

# Save the plot as an image
plot.save("diagram.png", dpi=300)