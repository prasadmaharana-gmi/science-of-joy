import streamlit as st
import pandas as pd
import plotly.express as px
import nltk
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide")

global_options = ['Liking', 'Meets Expectation', 'Product Improvement', 'All Questions']

#Setup side bar
with st.sidebar:
    selectbox_option = st.sidebar.selectbox(
    "Choose Question",
    ('Liking', 'Meets Expectation', 'Product Improvement', 'All Questions'))

st.title("Exploring Consumer Responses")

######### Top 10 words for entire dataset #######
st.subheader("Top 10 most frequent words used - Word Frequency")
#st.subheader("Chosen Question : "+str(selectbox_option))
top_10_df = pd.read_csv("C:\\Users\\g654674\\OneDrive - General Mills\\Work_OneDrive\\PGI\\Text Analytics\\app\\current_draft\\word_freq_samp_total.csv")
#max_freq = top_10_df['frequency'].max()
tot_samp1_df = top_10_df[(top_10_df['product'] == 'Chobani Original')  & (top_10_df['question'] == selectbox_option)]
tot_samp2_df = top_10_df[(top_10_df['product'] == 'Chobani Smooth')  & (top_10_df['question'] == selectbox_option)]

tot_fig = go.Figure(data=[
    go.Bar(name= tot_samp1_df['product'].unique()[0], x=tot_samp1_df['word'], y=tot_samp1_df['frequency'], text=tot_samp1_df['frequency']),
    go.Bar(name= tot_samp2_df['product'].unique()[0], x=tot_samp2_df['word'], y=tot_samp2_df['frequency'], text=tot_samp2_df['frequency']),
])
# Change the bar mode
tot_fig.update_layout(barmode='group',
xaxis={'categoryorder':'total descending'},
yaxis=dict(
    title='<b>Word Frequency</b>',
    titlefont_size=20,  
),
title = "Chosen Question : " + str(selectbox_option),
plot_bgcolor = "white",
font = dict(color = "#000000", size= 16, family="sans serif", ),)
st.plotly_chart(tot_fig, use_container_width=True)
#################################################


# ############# Table for for most used words
display = top_10_df['word'].unique()
new_arr = np.concatenate( (display, ["<All Words>"] ) )
options = list(range(len(new_arr)))
st.subheader("Tabular view for top 10 words with sentiment (sentiment score is indicative), choose word :")
top_10_selected_value = st.selectbox("", options, format_func=lambda x: new_arr[x])

word_choice = new_arr[top_10_selected_value]

combd_df_main = pd.read_csv("C:\\Users\\g654674\\OneDrive - General Mills\\Work_OneDrive\\PGI\\Text Analytics\\app\\current_draft\\combined_all_word_filtering.csv")

if word_choice != '<All Words>':
    combd_df = combd_df_main[(combd_df_main['Question'] == selectbox_option)  & (combd_df_main['lemm'].str.contains(str(word_choice)))]
else:
    combd_df = combd_df_main[(combd_df_main['Question'] == selectbox_option)]

table_fig = go.Figure(data=[go.Table(
columnwidth = [40,40,100,30,30],
header=dict(values=['Sample Name','Question Type','Response', 'Response Value', 'Sentiment Score'],
            line_color='darkslategray',
            align='left',
            font_size = 18),
cells=dict(values=[combd_df['product'].astype(str), combd_df['Question'].astype(str), combd_df['Response'].astype(str), combd_df['Response value'].fillna(0).astype(int).astype(str), round(combd_df['Sentiment Score'].fillna(0)).astype(int).astype(str) ],        
            align='left',
            # fill = dict(color='rgb(245,245,245)'),
            fill_color=['rgb(250,250,250)','rgb(250,250,250)','rgb(250,250,250)','rgb(250,250,250)',['rgb(128, 204, 51)' if val >= 60 else 'rgb(250,250,250)' for val in round(combd_df['Sentiment Score'].fillna(0)).astype(int)]],               
            # fill_color=['rgb(250,250,250)','rgb(250,250,250)','rgb(250,250,250)','rgb(250,250,250)',['rgb(255, 255, 0)' if val <= -30 else 'rgb(89, 204, 51)' if val <= 60 else 'rgb(89, 204, 51)' for val in round(combd_df['Sentiment Score'].fillna(0)).astype(int)]],               
            line_color='darkslategray',
            font_size = 16,height=30))
])

table_fig.update_layout(
title = "Chosen Question : " + str(selectbox_option) + " <br>Chosen Word : " + str(word_choice),
font = dict(color = "#000000", size= 16,family="sans serif",), )

st.plotly_chart(table_fig, use_container_width=True)
#################################################


#################################### Table for Collocation ########################################
#Collocation etc
st.subheader('Collocation - Words that appear frequently')
words = st.slider('Choose number of words', 2,3)

coll_df = combd_df_main[(combd_df_main['Question'] == selectbox_option)]
coll_df['lemm_clean'] = coll_df['lemm'].str.replace("\'|\[|\]|\,",'',regex=True)
coll_df['lemm_clean'] = coll_df['lemm_clean'].astype(str)

word_filter = lambda *w: str(word_choice) not in w

combined_ls = []
for id, rows in coll_df['lemm_clean'].iteritems():
    for items in rows.split(" "):        
        combined_ls.append(str(items))

text = nltk.Text(combined_ls)
if words == 2:
    finder = nltk.collocations.BigramCollocationFinder.from_words(text)
    finder.apply_freq_filter(2)
    if word_choice != '<All Words>':
        finder.apply_ngram_filter(word_filter)
elif words == 3:
    finder = nltk.collocations.TrigramCollocationFinder.from_words(text)
    finder.apply_freq_filter(2)
    if word_choice != '<All Words>':
        finder.apply_ngram_filter(word_filter)
df = pd.DataFrame(finder.ngram_fd.most_common(10), columns=['Words','Frequency'])
df['Words'] = df['Words'].astype(str)
df['Words'].replace(',',' -',regex=True, inplace = True)
df['Words'].replace("\(|\'|\)",'',regex=True, inplace = True)

coll_table_fig = go.Figure(data=[go.Table(
columnwidth = [100,100],
header=dict(values=['Words','Frequency'],        
            align='left',
            line_color='darkslategray',
            font_size = 18,),
cells=dict(values=[df['Words'].astype(str), df['Frequency'].astype(str), ],            
            align='left',
            fill = dict(color='rgb(250,250,250)'),
            line_color='darkslategray',     
            font_size = 16,
            height = 30))
])

coll_table_fig.update_layout(
title = "Chosen Question : " + str(selectbox_option) + " <br>Chosen Word : " + str(word_choice),
font = dict(color = "#000000", size= 16,family="sans serif",), )

st.plotly_chart(coll_table_fig, use_container_width=True)

###############################

#############Sentiment, Question + word wise##################
st.subheader("Top 10 most frequent words used - Sentiment Average")
#st.subheader("Chosen Question : " + str(selectbox_option))
sen_tot_fig = go.Figure(data=[
    go.Bar(name= tot_samp1_df['product'].unique()[0], x=tot_samp1_df['word'], y=tot_samp1_df['sentiment_mean'], text=round(tot_samp1_df['sentiment_mean'])),
    go.Bar(name= tot_samp2_df['product'].unique()[0], x=tot_samp2_df['word'], y=tot_samp2_df['sentiment_mean'], text=round(tot_samp2_df['sentiment_mean'])),
])
# Change the bar mode
sen_tot_fig.update_layout(barmode='group',
xaxis={'categoryorder':'total descending'},
yaxis=dict(
    title='<b>Sentiment Average</b>',
    titlefont_size=20,  
),
title = "Chosen Question : " + str(selectbox_option),
plot_bgcolor = "white",
font = dict(color = "#000000", size= 16,family="sans serif",), )
st.plotly_chart(sen_tot_fig, use_container_width=True)


############################# Scatter ####################

output_df = pd.read_csv("C:\\Users\\g654674\\OneDrive - General Mills\\Work_OneDrive\\PGI\\Text Analytics\\app\\output.csv")
st.subheader('Sentiment - Question')

if selectbox_option == 'Liking':
    output_df['like_compound_scaled'] = round((output_df['like_compound'] - output_df['like_compound'].min()) / (output_df['like_compound'].max() - output_df['like_compound'].min())*100)
    scatter_fig = px.scatter(output_df, x="like_compound", y="liking", size="like_compound_scaled", color="product_code", hover_data=['liking_free_text'])
    scatter_fig.update_layout(plot_bgcolor = "white",
                    font = dict(color = "#000000", size= 16,family="sans serif",),
                    title = "Chosen Question : " + str(selectbox_option),
                    xaxis = dict(title = "Sentiment", linecolor = "#909497"),
                    yaxis=dict(title=f'<b>{selectbox_option}</b>',titlefont_size=20, linecolor = "#909497" ),
                    legend=dict(title="")
                    )    
    scatter_fig.update_yaxes(range=[0,10], dtick=1,visible=True, showticklabels=True)  
    st.plotly_chart(scatter_fig, use_container_width=True)

elif selectbox_option == 'Meets Expectation':
    output_df['me_compound_scaled'] = round((output_df['meets_expectation_compound'] - output_df['meets_expectation_compound'].min()) / (output_df['meets_expectation_compound'].max() - output_df['meets_expectation_compound'].min())*100)
    scatter_fig = px.scatter(output_df, x="meets_expectation_compound", y="meets_expectation", size="me_compound_scaled", color="product_code", hover_data=['meets_expectation_free_text'])

    scatter_fig.update_layout(plot_bgcolor = "white",
                    font = dict(color = "#000000", size= 16,family="sans serif",),
                    title = "Chosen Question : " + str(selectbox_option),
                    xaxis = dict(title = "Sentiment", linecolor = "#909497"),
                    yaxis=dict(title=f'<b>{selectbox_option}</b>',titlefont_size=20,  linecolor = "#909497" ),
                    legend=dict(title="")
                    )    
    scatter_fig.update_yaxes(range=[0,10], dtick=1,visible=True, showticklabels=True)  
    st.plotly_chart(scatter_fig, use_container_width=True)

else :
    st.subheader("Cannot be generated for this selection")

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 


# import streamlit as st
# #from pre_process_functions import clean_data
# import pandas as pd
# import nltk

# st.set_page_config(layout="wide")


# st.header('Exploring Text Data')
# ######### Top 10 words for entire dataset #######
# top_10_df = pd.read_csv("C:\\Users\\g654674\\OneDrive - General Mills\\Work_OneDrive\\PGI\\Text Analytics\\app\\combined_samp_wise.csv")
# max_freq = top_10_df['frequency'].max()
# tot_samp1_df = top_10_df[top_10_df['product'] == 'Chobani Original']
# tot_samp2_df = top_10_df[top_10_df['product'] == 'Chobani Smooth']


# import plotly.graph_objects as go

# tot_fig = go.Figure(data=[
#     go.Bar(name= tot_samp1_df['product'].unique()[0], x=tot_samp1_df['word'], y=tot_samp1_df['frequency'], text=tot_samp1_df['frequency']),
#     go.Bar(name= tot_samp2_df['product'].unique()[0], x=tot_samp2_df['word'], y=tot_samp2_df['frequency'], text=tot_samp2_df['frequency']),
# ])
# # Change the bar mode
# tot_fig.update_layout(barmode='group',
# xaxis={'categoryorder':'total descending'},
# title = dict(text = "Top 10 most frequent words for open ended questions"),
# plot_bgcolor = "white",
# font = dict(color = "#909497"))
# st.plotly_chart(tot_fig, use_container_width=True)
# #################################################


# ############# Table for for most used words
# display = top_10_df['word'].unique()
# words = top_10_df['word'].drop_duplicates()
# options = list(range(len(display)))
# value = st.selectbox("Top 10 Words", options, format_func=lambda x: display[x])

# # product = top_10_df['product'].drop_duplicates()
# # word_choice = st.sidebar.selectbox('Choose word:', words)
# # product_choice = st.sidebar.selectbox('Choose Sample', product)
# # question_choice = st.sidebar.selectbox('Choose Question', ['Liking', 'Meets Expectation', 'Product Improvement' ])

# word_choice = display[value]

# df = pd.read_csv("C:\\Users\\g654674\\OneDrive - General Mills\\Work_OneDrive\\PGI\\Text Analytics\\app\\output.csv")
# # Subset and union 
# liking_df = df[['product_code','liking', 'liking_free_text', 'like_compound', 'liking_free_text_no_punc_tokenized_no_stop_lemm']]
# liking_df["question"] = "Liking"
# meets_df = df[['product_code','meets_expectation', 'meets_expectation_free_text', 'meets_expectation_compound', 'meets_expectation_free_text_no_punc_tokenized_no_stop_lemm']]
# meets_df["question"] = "Meets Expectation"

# prod_df = df[['product_code','product_improvement_free_text', 'prod_improve_compound', 'product_improvement_free_text_no_punc_tokenized_no_stop_lemm']]
# prod_df["meets_expectation"] = None
# prod_df["question"] = "Product Improvement"
# prod_df = prod_df[['product_code','meets_expectation','product_improvement_free_text', 'prod_improve_compound', 'product_improvement_free_text_no_punc_tokenized_no_stop_lemm', 'question']]

# liking_df.columns = ['product','Response value','Response','Sentiment Score','lemm','Question']
# prod_df.columns = ['product','Response value','Response','Sentiment Score','lemm','Question']
# meets_df.columns = ['product','Response value','Response','Sentiment Score','lemm','Question']

# comb1_df = liking_df.append(meets_df)
# combd_df = comb1_df.append(prod_df)
# combd_df = comb1_df [['product','Question','Response','Sentiment Score', 'Response value', 'lemm']]
# combd_df['Sentiment Score'] = combd_df['Sentiment Score']*100
# #combd_df.to_csv("C:\\Users\\g654674\\OneDrive - General Mills\\Work_OneDrive\\PGI\\Text Analytics\\app\\combined_all.csv",index=False)
# # combd_df = combd_df.loc[(combd_df['Question'] == question_choice) & (combd_df['product'] == product_choice)]

# combd_df = combd_df[combd_df['lemm'].str.contains(str(word_choice))]

# fig = go.Figure(data=[go.Table(
#     columnwidth = [40,40,100,30,30],
#     header=dict(values=['Sample Name','Question Type','Response', 'Sentiment Score', 'Response Value'],
#                 #fill_color='paleturquoise',
#                 align='left',
#                 font_size = 15),
#     cells=dict(values=[combd_df['product'].astype(str), combd_df['Question'].astype(str), combd_df['Response'].astype(str), round(combd_df['Sentiment Score']).astype(int).astype(str), combd_df['Response value'].astype(str) ],
#                #fill_color='lavender',
#                align='left',
#                fill = dict(color='rgb(245,245,245)'),               
#             #  line_color='darkslategray',
#                font_size = 15))
# ])
# # go_fig = go.Figure()
# #fig.update_layout(width=100, height=900)
# # obj = go.Table(header = dict(values=["Countries", "Vlaue"]),
# #     cells = dict(values=[combd_df['product'].astype(str), combd_df['Response'].astype(str)]))

# # go_fig.add_trace(obj)               
# st.plotly_chart(fig, use_container_width=True)

# #st.dataframe(combd_df.style.set_properties(subset=['Response'], **{'width': '322200px'}))
# ###########################################

# import plotly.express as px
# from plotly.subplots import make_subplots
# import plotly.graph_objects as go


# clean_df = pd.read_csv("C:\\Users\\g654674\\OneDrive - General Mills\\Work_OneDrive\\PGI\\Text Analytics\\app\\samp_wise.csv")
# max_freq = clean_df['frequency'].max()
# samp1_df = clean_df[clean_df['product'] == 'Chobani Original']
# samp2_df = clean_df[clean_df['product'] == 'Chobani Smooth']

# def SetColor(y):
#     if(y<=-30):
#         return "darkred"
#     elif(y > -30 and y < 40):        
#         return "lightgreen"
#     elif(y > 40 and y < 60):
#         return "limegreen"        
#     elif(y > 60):
#         return "forestgreen"

# bar_fig = go.Figure()

# bar_fig.add_trace(go.Bar(name=samp1_df['product'].unique()[0], x=clean_df['word'], y=samp1_df['sentiment_mean'],
#                      text=samp1_df['product'].unique()[0],
#                      hovertemplate = "<br>Sentiment Average: %{text}",
#                      marker=dict(color = list(map(SetColor, samp1_df['sentiment_mean'])))))
# bar_fig.add_trace(go.Bar(name=samp2_df['product'].unique()[0], x=clean_df['word'], y=samp2_df['sentiment_mean'],
#                      text=round(samp2_df['sentiment_mean'],0),
#                      hovertemplate = "<br>Sentiment Average: %{text}",
#                      marker=dict(color = list(map(SetColor, samp2_df['sentiment_mean'])))))

# # fig = go.Figure(data=[
# #     go.Bar(name=samp1_df['product'].unique()[0], x=clean_df['word'], y=samp1_df['sentiment_mean']),
# #     go.Bar(name=samp2_df['product'].unique()[0], x=clean_df['word'], y=samp2_df['sentiment_mean'])
# # ])
# # Change the bar mode
# bar_fig.update_layout(barmode='group',showlegend = False, plot_bgcolor = "white",title = dict(text = " Sample Wise sentiment comparison - Chobani Original (Left) vs Chobani Smooth (Right)"))
# st.plotly_chart(bar_fig, use_container_width=True)

# # figs = px.histogram(df, x=clean_df['word'], y=clean_df['sentiment_mean'],
# #              color=clean_df['sentiment_mean'], barmode='group')             
# # st.plotly_chart(figs, use_container_width=True)

# # figure = make_subplots(rows=1, cols=2,shared_yaxes=True, shared_xaxes=False, subplot_titles=(samp1_df['product'].unique()[0],samp2_df['product'].unique()[0]))
# # figure.add_trace(go.Bar(
# # 		            x=samp1_df['frequency'],
# # 		            y=samp1_df['word'],
# #                     name = samp1_df['product'].unique()[0],                    
# #                     text=round(samp1_df['sentiment_mean']),
# #                     textfont=dict(color="rgba(0,0,0,0)", size=1),
# #                     hovertemplate = "<br>Frequency: %{x} <br>Sentiment Average: %{text}",
# #                     #textposition= 'outside',
# #                     marker=dict(color = list(map(SetColor, samp1_df['sentiment_mean']))),
# #                     orientation='h'),
# #                     row = 1, 
# #                     col = 1,
# #                     )
# # figure.add_trace(go.Bar(
# # 	            x=samp2_df['frequency'],
# # 		        y=samp2_df['word'],
# #                 name = samp2_df['product'].unique()[0],
# #                 text=round(samp2_df['sentiment_mean']),
# #                 textfont=dict(color="rgba(0,0,0,0)", size=1),  
# #                 #textposition = 'outside',
# #                 hovertemplate = "<br>Frequency: %{x} <br>Sentiment Average: %{text}",
# #                 marker=dict(color = list(map(SetColor, samp2_df['sentiment_mean']))),
# #                 orientation='h'
# #                 ),                
# # 				row = 1, 
# # 				col = 2)

# # figure.update_yaxes(autorange="reversed")

# # figure.update_xaxes(range=[1,max_freq+5], dtick=5,visible=True, showticklabels=True)
# # figure.update_xaxes(title_text="", row=1, col=1)
# # figure.update_xaxes(title_text="", row=1, col=2)

# # figure.update_layout(plot_bgcolor = "white",
# #                     font = dict(color = "#909497"),
# #                     title = dict(text = "Top 10 Most Used Words with Average Sentiment"),
# #                     # xaxis = dict(title = "Word Frequency", linecolor = "#909497"), 
# #                     yaxis = dict(title = "Words", linecolor = "#909497" ),
# #                     showlegend = False,
# #                     # hovermode="x unified" ,
# #                     height=400, width=200
# #                     ) #apply our custom category order

# # figure = px.bar(samp1_df,x='word',y='sentiment_mean', color='sentiment_mean', barmode='stack', orientation = 'v')
# # figure2 = px.bar(samp2_df,x='word',y='sentiment_mean', color='sentiment_mean', barmode='stack', orientation = 'v')

# # st.plotly_chart(figure, use_container_width=True)
# # st.plotly_chart(figure2, use_container_width=True)

# hide_streamlit_style = """
#             <style>
#             #MainMenu {visibility: hidden;}
#             footer {visibility: hidden;}
#             </style>
#             """
# st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

# ##Concordance 
# distinct_word_list = clean_df['word'].unique().tolist()


# output_df = pd.read_csv("C:\\Users\\g654674\\OneDrive - General Mills\\Work_OneDrive\\PGI\\Text Analytics\\app\\output.csv")
# st.subheader('Sentiment - Question')
# option = st.selectbox(
#      'Choose Question',
#      ('Liking', 'Meets expectation'))

# if option == 'Liking':
#     output_df['like_compound_scaled'] = round((output_df['like_compound'] - output_df['like_compound'].min()) / (output_df['like_compound'].max() - output_df['like_compound'].min())*100)
#     scatter_fig = px.scatter(output_df, x="like_compound", y="liking", size="like_compound_scaled", color="product_code", hover_data=['liking_free_text'])

#     scatter_fig.update_layout(plot_bgcolor = "white",
#                     font = dict(color = "#909497"),
#                     title = dict(text = "Sentiment/Liking"),
#                     xaxis = dict(title = "Sentiment", linecolor = "#909497"),
#                     yaxis = dict(title = "Liking", linecolor = "#909497" )
#                     )
#     scatter_fig.update_layout(
#     legend=dict(
#         title="",
#         font=dict(
#             family="Courier",
#             size=12,
#             color="#909497"
#         ),
#         bordercolor="#909497",
#         borderwidth=1
#     )
#     )
#     scatter_fig.update_yaxes(range=[0,10], dtick=1,visible=True, showticklabels=True)  
#     #scatter_fig.update_xaxes(range=[0,110], dtick=10,visible=True, showticklabels=True)
#     #scatter_fig.update_xaxes(visible=True, showticklabels=True)
#     st.plotly_chart(scatter_fig, use_container_width=True)

# elif option == 'Meets expectation':
#     output_df['meets_expectation_compound_scaled'] = round((output_df['meets_expectation_compound'] - output_df['meets_expectation_compound'].min()) / (output_df['meets_expectation_compound'].max() - output_df['meets_expectation_compound'].min())*100)
#     scatter_fig = px.scatter(output_df, x="meets_expectation", y="meets_expectation_compound_scaled", size="meets_expectation_compound_scaled", color="product_code", hover_data=['meets_expectation_free_text'])
#     scatter_fig.update_layout(plot_bgcolor = "white",
#                     font = dict(color = "#909497"),
#                     title = dict(text = "Sentiment/Meets Expectation"),
#                     yaxis = dict(title = "Sentiment (Scaled)", linecolor = "#909497"),
#                     xaxis = dict(title = "Meets Expectation", linecolor = "#909497")                        
#                     )
#     scatter_fig.update_layout(
#     legend=dict(        
#         title="",        
#         font=dict(
#             family="Courier",
#             size=12,
#             color="#909497"
#         ),        
#         bordercolor="#909497",
#         borderwidth=1
#     )
#     )
#     scatter_fig.update_yaxes(range=[0,110], dtick=10,visible=True, showticklabels=True)     
#     #scatter_fig.update_xaxes(visible=True, showticklabels=True)                
#     st.plotly_chart(scatter_fig, use_container_width=True)


# #pyvis
# #graph viz

# #Collocation etc
# st.subheader('Collocation - Two or more words that tend to appear frequently together')
# words = st.slider('Choose number of words', 2,3)

# coll_df = pd.read_csv("C:\\Users\\g654674\\OneDrive - General Mills\\Work_OneDrive\\PGI\\Text Analytics\\app\\current_draft\\combined_all_word_filtering.csv")
# coll_df['lemm_clean'] = coll_df['lemm'].str.replace("\'|\[|\]|\,",'',regex=True)
# coll_df['lemm_clean'] = coll_df['lemm_clean'].astype(str)

# combined_ls = []
# for id, rows in coll_df['combined_no_punc_tokenized_no_stop_lemm'].iteritems():
#     for items in rows.split(" "):
#         combined_ls.append(items)

# text = nltk.Text(combined_ls)
# if words == 2:
#     finder = nltk.collocations.BigramCollocationFinder.from_words(text)
# elif words == 3:
#     finder = nltk.collocations.TrigramCollocationFinder.from_words(text)
# df = pd.DataFrame(finder.ngram_fd.most_common(10), columns=['Words','Frequency'])
# st.table(df)

# # text = nltk.Text(combined_ls)
# # text.concordance("like", lines=1)