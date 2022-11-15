import pandas as pd

# lst=[1,2,3]
# class Car:
#     def __init__(self,lst:list)->list:
#         self.lst=lst
#
#     def concat(self,inp:list)->list:
#         conc=self.lst[0]+self.lst[1]
#         return conc,self.lst

a=[[1,2],[2,3]]
class Feature_engineering:
    """Adding inderect features"""
    def __init__(self,df):
        self.df=df
    def ret(self):
        return self.df

    def adding_inderect_features(self):
        df = self.df.to_frame(name='comment_text')
        df.loc[:, 'count_word'] = df["comment_text"].apply(lambda x: len(str(x).split()))
        df.loc[:, 'count_unique_word'] = df["comment_text"].apply(lambda x: len(set(str(x).split())))
        df.loc[:, 'count_letters'] = df["comment_text"].apply(lambda x: len(str(x)))
        df.loc[:, "count_punctuations"] = df["comment_text"].apply(
            lambda x: len([c for c in str(x) if c in string.punctuation]))
        df.loc[:, "count_words_upper"] = df["comment_text"].apply(
            lambda x: len([w for w in str(x).split() if w.isupper()]))
        df.loc[:, "count_words_title"] = df["comment_text"].apply(
            lambda x: len([w for w in str(x).split() if w.istitle()]))
        df.loc[:, "count_stopwords"] = df["comment_text"].apply(
            lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))
        df.loc[:, "mean_word_len"] = df["comment_text"].apply(
            lambda x: round(np.mean([len(w) for w in str(x).split()]), 2))
        df.loc[:, 'word_unique_percent'] = df.loc[:, 'count_unique_word'] * 100 / df['count_word']
        df.loc[:, 'punct_percent'] = df.loc[:, 'count_punctuations'] * 100 / df['count_word']

        self.df = df


def adding_inderect_features2(df_in):
    df = df_in.to_frame(name='comment_text')
    df.loc[:, 'count_word'] = df["comment_text"].apply(lambda x: len(str(x).split()))
    df.loc[:, 'count_unique_word'] = df["comment_text"].apply(lambda x: len(set(str(x).split())))
    df.loc[:, 'count_letters'] = df["comment_text"].apply(lambda x: len(str(x)))
    df.loc[:, "count_punctuations"] = df["comment_text"].apply(
        lambda x: len([c for c in str(x) if c in string.punctuation]))
    df.loc[:, "count_words_upper"] = df["comment_text"].apply(
        lambda x: len([w for w in str(x).split() if w.isupper()]))
    df.loc[:, "count_words_title"] = df["comment_text"].apply(
        lambda x: len([w for w in str(x).split() if w.istitle()]))
    df.loc[:, "count_stopwords"] = df["comment_text"].apply(
        lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))
    df.loc[:, "mean_word_len"] = df["comment_text"].apply(
        lambda x: round(np.mean([len(w) for w in str(x).split()]), 2))
    df.loc[:, 'word_unique_percent'] = df.loc[:, 'count_unique_word'] * 100 / df['count_word']
    df.loc[:, 'punct_percent'] = df.loc[:, 'count_punctuations'] * 100 / df['count_word']

    return df
feature_eng = Feature_engineering(df=pd.DataFrame({"text":"text"}))
feature_eng.df
feature_eng.adding_inderect_features()
feature_eng.df
feature_eng.adding_inderect_features2()


new_df = adding_inderect_features2(df_in=pd.DataFrame({"text":"text"}))
new_df2 = adding_inderect_features2(df_in=new_df)

