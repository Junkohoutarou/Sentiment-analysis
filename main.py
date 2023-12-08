import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt
import seaborn as sns

st.title('Data visualization')
st.header('Upload data file')
data_file = st.file_uploader('Choose a csv file', type=(['.csv']))

if data_file is not None:
    df = pd.read_csv(data_file)

    st.header('Show data')
    st.dataframe(df)

    st.header('Descriptive statistics')
    st.table(df.describe())

    st.header('Show data information')
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())

    st.header('Visualize each attribute')
    for col in df.columns:
        fig, ax = plt.subplots()
        ax.hist(df[col], bins=20)  
        plt.xlabel(col)
        plt.ylabel('Quantity')
        st.pyplot(fig)

    st.header('Show correlation between variables')
    
    # Xóa các dòng chứa giá trị NaN trước khi tính toán ma trận tương quan
    df_corr = df.dropna()

    # Kiểm tra xem DataFrame có đủ dữ liệu để tính toán không
    if len(df_corr) >= 2 and len(df_corr.columns) >= 2 and df_corr.select_dtypes(include='number').shape[1] >= 2:
        fig, ax = plt.subplots()
        sns.heatmap(df_corr.corr(method='pearson'), ax=ax, vmax=1, square=True, annot=True, cmap='Reds')
        st.write(fig)
    else:
        st.warning('Not enough numeric data to calculate correlation matrix.')

    output = st.radio('Choose a dependent variable', df.columns)
    st.header('Show correlation between variables')
    for col in list(df.columns):
        if col != output:
            fig, ax = plt.subplots()
            ax.scatter(x=df[col], y=df[output])
            plt.xlabel(col)
            plt.ylabel(output)
            st.pyplot(fig)
