{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## importing the relevant modules for our project"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import datetime\n",
    "import re\n",
    "from scipy.sparse import hstack, csr_matrix\n",
    "\n",
    "import nltk\n",
    "from nltk.corpus import stopwords \n",
    "from nltk.tokenize import word_tokenize\n",
    "from nltk.stem import WordNetLemmatizer, PorterStemmer\n",
    "\n",
    "from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer\n",
    "from sklearn.preprocessing import LabelEncoder, StandardScaler\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.decomposition import TruncatedSVD\n",
    "from sklearn.svm import LinearSVR\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "from sklearn.metrics import mean_absolute_error\n",
    "\n",
    "from wordcloud import WordCloud, STOPWORDS \n",
    "from textblob import TextBlob\n",
    "import lightgbm as lgb\n",
    "\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "# !pip install wordcloud    ## installing wordclaud & lightgbm because this module was installed\n",
    "# !pip install lightgbm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "C:\\Users\\shaqu\\Desktop\\Data Science Projects\\predict the news sentiment ~ ZS\\news sentiment\\dataset\n"
     ]
    }
   ],
   "source": [
    "\n",
    "cd C:\\Users\\shaqu\\Desktop\\Data Science Projects\\predict the news sentiment ~ ZS\\news sentiment\\dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "# reading the dataset!!\n",
    "\n",
    "train = pd.read_csv('train_file.csv')\n",
    "test = pd.read_csv('test_file.csv')\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## data exploration!!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>IDLink</th>\n",
       "      <th>Title</th>\n",
       "      <th>Headline</th>\n",
       "      <th>Source</th>\n",
       "      <th>Topic</th>\n",
       "      <th>PublishDate</th>\n",
       "      <th>Facebook</th>\n",
       "      <th>GooglePlus</th>\n",
       "      <th>LinkedIn</th>\n",
       "      <th>SentimentTitle</th>\n",
       "      <th>SentimentHeadline</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>0</td>\n",
       "      <td>Tr3CMgRv1N</td>\n",
       "      <td>Obama Lays Wreath at Arlington National Cemetery</td>\n",
       "      <td>Obama Lays Wreath at Arlington National Cemete...</td>\n",
       "      <td>USA TODAY</td>\n",
       "      <td>obama</td>\n",
       "      <td>2002-04-02 00:00:00</td>\n",
       "      <td>-1</td>\n",
       "      <td>-1</td>\n",
       "      <td>-1</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>-0.053300</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1</td>\n",
       "      <td>Wc81vGp8qZ</td>\n",
       "      <td>A Look at the Health of the Chinese Economy</td>\n",
       "      <td>Tim Haywood, investment director business-unit...</td>\n",
       "      <td>Bloomberg</td>\n",
       "      <td>economy</td>\n",
       "      <td>2008-09-20 00:00:00</td>\n",
       "      <td>-1</td>\n",
       "      <td>-1</td>\n",
       "      <td>-1</td>\n",
       "      <td>0.208333</td>\n",
       "      <td>-0.156386</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2</td>\n",
       "      <td>zNGH03CrZH</td>\n",
       "      <td>Nouriel Roubini: Global Economy Not Back to 2008</td>\n",
       "      <td>Nouriel Roubini, NYU professor and chairman at...</td>\n",
       "      <td>Bloomberg</td>\n",
       "      <td>economy</td>\n",
       "      <td>2012-01-28 00:00:00</td>\n",
       "      <td>-1</td>\n",
       "      <td>-1</td>\n",
       "      <td>-1</td>\n",
       "      <td>-0.425210</td>\n",
       "      <td>0.139754</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3</td>\n",
       "      <td>3sM1H0W8ts</td>\n",
       "      <td>Finland GDP Expands In Q4</td>\n",
       "      <td>Finland's economy expanded marginally in the t...</td>\n",
       "      <td>RTT News</td>\n",
       "      <td>economy</td>\n",
       "      <td>2015-03-01 00:06:00</td>\n",
       "      <td>-1</td>\n",
       "      <td>-1</td>\n",
       "      <td>-1</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.026064</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4</td>\n",
       "      <td>wUbnxgvqaZ</td>\n",
       "      <td>Tourism, govt spending buoys Thai economy in J...</td>\n",
       "      <td>Tourism and public spending continued to boost...</td>\n",
       "      <td>The Nation - Thailand&amp;#39;s English news</td>\n",
       "      <td>economy</td>\n",
       "      <td>2015-03-01 00:11:00</td>\n",
       "      <td>-1</td>\n",
       "      <td>-1</td>\n",
       "      <td>-1</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.141084</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       IDLink                                              Title  \\\n",
       "0  Tr3CMgRv1N   Obama Lays Wreath at Arlington National Cemetery   \n",
       "1  Wc81vGp8qZ        A Look at the Health of the Chinese Economy   \n",
       "2  zNGH03CrZH   Nouriel Roubini: Global Economy Not Back to 2008   \n",
       "3  3sM1H0W8ts                          Finland GDP Expands In Q4   \n",
       "4  wUbnxgvqaZ  Tourism, govt spending buoys Thai economy in J...   \n",
       "\n",
       "                                            Headline  \\\n",
       "0  Obama Lays Wreath at Arlington National Cemete...   \n",
       "1  Tim Haywood, investment director business-unit...   \n",
       "2  Nouriel Roubini, NYU professor and chairman at...   \n",
       "3  Finland's economy expanded marginally in the t...   \n",
       "4  Tourism and public spending continued to boost...   \n",
       "\n",
       "                                     Source    Topic          PublishDate  \\\n",
       "0                                 USA TODAY    obama  2002-04-02 00:00:00   \n",
       "1                                 Bloomberg  economy  2008-09-20 00:00:00   \n",
       "2                                 Bloomberg  economy  2012-01-28 00:00:00   \n",
       "3                                  RTT News  economy  2015-03-01 00:06:00   \n",
       "4  The Nation - Thailand&#39;s English news  economy  2015-03-01 00:11:00   \n",
       "\n",
       "   Facebook  GooglePlus  LinkedIn  SentimentTitle  SentimentHeadline  \n",
       "0        -1          -1        -1        0.000000          -0.053300  \n",
       "1        -1          -1        -1        0.208333          -0.156386  \n",
       "2        -1          -1        -1       -0.425210           0.139754  \n",
       "3        -1          -1        -1        0.000000           0.026064  \n",
       "4        -1          -1        -1        0.000000           0.141084  "
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.head()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(55932, 11)"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>IDLink</th>\n",
       "      <th>Title</th>\n",
       "      <th>Headline</th>\n",
       "      <th>Source</th>\n",
       "      <th>Topic</th>\n",
       "      <th>PublishDate</th>\n",
       "      <th>Facebook</th>\n",
       "      <th>GooglePlus</th>\n",
       "      <th>LinkedIn</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>0</td>\n",
       "      <td>tFrqIR6Chj</td>\n",
       "      <td>Sliding Economy: FG fights back with N3trn TSA...</td>\n",
       "      <td>With the 2016 budget now passed by the Nationa...</td>\n",
       "      <td>BusinessDay</td>\n",
       "      <td>economy</td>\n",
       "      <td>2016-03-29 01:41:12</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1</td>\n",
       "      <td>DVAaGErjlF</td>\n",
       "      <td>Microsoft shows how HoloLens can bring distant...</td>\n",
       "      <td>A recent Microsoft Research video shows how th...</td>\n",
       "      <td>Daily Mail</td>\n",
       "      <td>microsoft</td>\n",
       "      <td>2016-03-29 01:41:27</td>\n",
       "      <td>121</td>\n",
       "      <td>2</td>\n",
       "      <td>13</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2</td>\n",
       "      <td>OT9UIZm5M2</td>\n",
       "      <td>Microsoft’s Twitter Robot Praises Hitler, Trum...</td>\n",
       "      <td>* Microsoft teamed with Bing to create TayTwee...</td>\n",
       "      <td>EURweb</td>\n",
       "      <td>microsoft</td>\n",
       "      <td>2016-03-29 01:47:00</td>\n",
       "      <td>12</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3</td>\n",
       "      <td>lflGp3q2Fj</td>\n",
       "      <td>Flood of Central Bank Moves Can't Get World Ec...</td>\n",
       "      <td>Central bankers have managed to steer the worl...</td>\n",
       "      <td>Bloomberg via Yahoo! Finance</td>\n",
       "      <td>economy</td>\n",
       "      <td>2016-03-29 02:00:00</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4</td>\n",
       "      <td>zDYG0SoovZ</td>\n",
       "      <td>USD/JPY: bears lining up on mixed U.S. economy...</td>\n",
       "      <td>However, this streak of seven-day gains might ...</td>\n",
       "      <td>FXStreet</td>\n",
       "      <td>economy</td>\n",
       "      <td>2016-03-29 02:01:07</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       IDLink                                              Title  \\\n",
       "0  tFrqIR6Chj  Sliding Economy: FG fights back with N3trn TSA...   \n",
       "1  DVAaGErjlF  Microsoft shows how HoloLens can bring distant...   \n",
       "2  OT9UIZm5M2  Microsoft’s Twitter Robot Praises Hitler, Trum...   \n",
       "3  lflGp3q2Fj  Flood of Central Bank Moves Can't Get World Ec...   \n",
       "4  zDYG0SoovZ  USD/JPY: bears lining up on mixed U.S. economy...   \n",
       "\n",
       "                                            Headline  \\\n",
       "0  With the 2016 budget now passed by the Nationa...   \n",
       "1  A recent Microsoft Research video shows how th...   \n",
       "2  * Microsoft teamed with Bing to create TayTwee...   \n",
       "3  Central bankers have managed to steer the worl...   \n",
       "4  However, this streak of seven-day gains might ...   \n",
       "\n",
       "                         Source      Topic          PublishDate  Facebook  \\\n",
       "0                   BusinessDay    economy  2016-03-29 01:41:12         0   \n",
       "1                    Daily Mail  microsoft  2016-03-29 01:41:27       121   \n",
       "2                        EURweb  microsoft  2016-03-29 01:47:00        12   \n",
       "3  Bloomberg via Yahoo! Finance    economy  2016-03-29 02:00:00         0   \n",
       "4                      FXStreet    economy  2016-03-29 02:01:07         3   \n",
       "\n",
       "   GooglePlus  LinkedIn  \n",
       "0           0         1  \n",
       "1           2        13  \n",
       "2           1         0  \n",
       "3           0         3  \n",
       "4           0         0  "
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(37288, 9)"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "## now focussing on our training dataset first!!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 55932 entries, 0 to 55931\n",
      "Data columns (total 11 columns):\n",
      "IDLink               55932 non-null object\n",
      "Title                55932 non-null object\n",
      "Headline             55932 non-null object\n",
      "Source               55757 non-null object\n",
      "Topic                55932 non-null object\n",
      "PublishDate          55932 non-null object\n",
      "Facebook             55932 non-null int64\n",
      "GooglePlus           55932 non-null int64\n",
      "LinkedIn             55932 non-null int64\n",
      "SentimentTitle       55932 non-null float64\n",
      "SentimentHeadline    55932 non-null float64\n",
      "dtypes: float64(2), int64(3), object(6)\n",
      "memory usage: 4.7+ MB\n"
     ]
    }
   ],
   "source": [
    "train.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "# columns ['IDLink', 'Title', 'Headline', 'Source', 'Topic', 'PublishDate'] are object type like string and columns ['Facebook', 'GooglePlus', 'LinkedIn',] are numeric type integers and columns ['SentimentTitle',\n",
    "#        'SentimentHeadline'] are numeric type float !!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Facebook</th>\n",
       "      <th>GooglePlus</th>\n",
       "      <th>LinkedIn</th>\n",
       "      <th>SentimentTitle</th>\n",
       "      <th>SentimentHeadline</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>count</td>\n",
       "      <td>55932.000000</td>\n",
       "      <td>55932.000000</td>\n",
       "      <td>55932.000000</td>\n",
       "      <td>55932.000000</td>\n",
       "      <td>55932.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>mean</td>\n",
       "      <td>132.050329</td>\n",
       "      <td>4.551616</td>\n",
       "      <td>14.300132</td>\n",
       "      <td>-0.006318</td>\n",
       "      <td>-0.029577</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>std</td>\n",
       "      <td>722.931314</td>\n",
       "      <td>21.137177</td>\n",
       "      <td>76.651420</td>\n",
       "      <td>0.137569</td>\n",
       "      <td>0.143038</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>min</td>\n",
       "      <td>-1.000000</td>\n",
       "      <td>-1.000000</td>\n",
       "      <td>-1.000000</td>\n",
       "      <td>-0.838525</td>\n",
       "      <td>-0.755355</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>25%</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>-0.079057</td>\n",
       "      <td>-0.116927</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>50%</td>\n",
       "      <td>6.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>-0.027277</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>75%</td>\n",
       "      <td>37.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>0.063969</td>\n",
       "      <td>0.057354</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>max</td>\n",
       "      <td>49211.000000</td>\n",
       "      <td>1267.000000</td>\n",
       "      <td>3716.000000</td>\n",
       "      <td>0.962354</td>\n",
       "      <td>0.964646</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "           Facebook    GooglePlus      LinkedIn  SentimentTitle  \\\n",
       "count  55932.000000  55932.000000  55932.000000    55932.000000   \n",
       "mean     132.050329      4.551616     14.300132       -0.006318   \n",
       "std      722.931314     21.137177     76.651420        0.137569   \n",
       "min       -1.000000     -1.000000     -1.000000       -0.838525   \n",
       "25%        0.000000      0.000000      0.000000       -0.079057   \n",
       "50%        6.000000      0.000000      0.000000        0.000000   \n",
       "75%       37.000000      2.000000      4.000000        0.063969   \n",
       "max    49211.000000   1267.000000   3716.000000        0.962354   \n",
       "\n",
       "       SentimentHeadline  \n",
       "count       55932.000000  \n",
       "mean           -0.029577  \n",
       "std             0.143038  \n",
       "min            -0.755355  \n",
       "25%            -0.116927  \n",
       "50%            -0.027277  \n",
       "75%             0.057354  \n",
       "max             0.964646  "
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Checking the misssing null, values in the dataset!!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "IDLink                 0\n",
       "Title                  0\n",
       "Headline               0\n",
       "Source               175\n",
       "Topic                  0\n",
       "PublishDate            0\n",
       "Facebook               0\n",
       "GooglePlus             0\n",
       "LinkedIn               0\n",
       "SentimentTitle         0\n",
       "SentimentHeadline      0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.isnull().sum()\n",
    "\n",
    "# source column have 175 records as null!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Bloomberg           992\n",
       "Reuters             763\n",
       "ABC News            645\n",
       "New York Times      573\n",
       "The Guardian        551\n",
       "Business Insider    550\n",
       "Forbes              484\n",
       "Economic Times      461\n",
       "CNN                 447\n",
       "WinBeta             445\n",
       "Name: Source, dtype: int64"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train['Source'].value_counts()[:10]\n",
    "\n",
    "# As we can see that most of the source is from bloomberg company then we just fill the null values with bloomberg for easy processing of our data!!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "train['Source'] = train['Source'].fillna('Bloomberg')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "IDLink               0\n",
       "Title                0\n",
       "Headline             0\n",
       "Source               0\n",
       "Topic                0\n",
       "PublishDate          0\n",
       "Facebook             0\n",
       "GooglePlus           0\n",
       "LinkedIn             0\n",
       "SentimentTitle       0\n",
       "SentimentHeadline    0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# checkiung the null values again!!\n",
    "train.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "## text column cleansing!!\n",
    "\n",
    "# remove the any alpha numerica nd spaces from our dataset to make it clean for our ML modelling!!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package stopwords to\n",
      "[nltk_data]     C:\\Users\\shaqu\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Package stopwords is already up-to-date!\n",
      "[nltk_data] Downloading package punkt to\n",
      "[nltk_data]     C:\\Users\\shaqu\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Package punkt is already up-to-date!\n",
      "[nltk_data] Downloading package wordnet to\n",
      "[nltk_data]     C:\\Users\\shaqu\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Package wordnet is already up-to-date!\n"
     ]
    }
   ],
   "source": [
    "nltk.download('stopwords')\n",
    "nltk.download('punkt')\n",
    "nltk.download('wordnet')\n",
    "\n",
    "stop = set(stopwords.words('english'))\n",
    "\n",
    "def clean(text):\n",
    "  text_token = word_tokenize(text)\n",
    "  filtered_text = ' '.join([w.lower() for w in text_token if w.lower() not in stop and len(w) > 2])\n",
    "  filtered_text = filtered_text.replace(r\"[^a-zA-Z]+\", '')\n",
    "  text_only = re.sub(r'\\b\\d+\\b', '', filtered_text)\n",
    "  clean_text = text_only.replace(',', '').replace('.', '').replace(':', '')\n",
    "  return clean_text"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "IDLink           0\n",
       "Title            0\n",
       "Headline         0\n",
       "Source         101\n",
       "Topic            0\n",
       "PublishDate      0\n",
       "Facebook         0\n",
       "GooglePlus       0\n",
       "LinkedIn         0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "## checking the null valus for test data\n",
    "test.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "test['Source'] = test['Source'].fillna('Bloomberg')\n",
    "\n",
    "# filling the values with the max value of source!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "train['Text_Title'] = train['Title'] + ' ' + train['Source'] + ' ' + train['Topic']\n",
    "test['Text_Title'] = test['Title'] + ' ' + test['Source'] + ' ' + test['Topic']\n",
    "\n",
    "train['Text_Headline'] = train['Headline'] + ' ' + train['Source'] + ' ' + train['Topic']\n",
    "test['Text_Headline'] = test['Headline'] + ' ' + test['Source'] + ' ' + test['Topic']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'Tourism, govt spending buoys Thai economy in January The Nation - Thailand&#39;s English news economy'"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train['Text_Title'][:10]\n",
    "train['Text_Title'][4]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "train['Text_Title'] = [clean(x) for x in train['Text_Title']]\n",
    "test['Text_Title'] = [clean(x) for x in test['Text_Title']]\n",
    "\n",
    "train['Text_Headline'] = [clean(x) for x in train['Text_Headline']]\n",
    "test['Text_Headline'] = [clean(x) for x in test['Text_Headline']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    obama lays wreath arlington national cemetery ...\n",
       "1        look health chinese economy bloomberg economy\n",
       "2    nouriel roubini global economy back  bloomberg...\n",
       "3                 finland gdp expands rtt news economy\n",
       "4    tourism govt spending buoys thai economy janua...\n",
       "5    intellitec solutions host 13th annual spring m...\n",
       "6                      monday feb  bloomberg palestine\n",
       "7    obama stars pay musical tribute ray charles co...\n",
       "8    fire claims -year-old barn hancock county wthr...\n",
       "9    microsoft new windows targets apple new kerala...\n",
       "Name: Text_Title, dtype: object"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train['Text_Title'][:10]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "# There is non iisue of whitespace problem!!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "## lets plot the visualization of the text title !!\n",
    "text = \"\"\n",
    "for ind, row in train.iterrows():\n",
    "    text += row[\"Text_Title\"] + \" \"\n",
    "text = text.strip()\n",
    "\n",
    "wordcloud = WordCloud(background_color='white', width=1000, height=600, max_font_size=300, max_words=50).generate(text)\n",
    "wordcloud.recolor(random_state=ind*312)\n",
    "plt.imshow(wordcloud)\n",
    "plt.axis(\"off\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# plot the text headlines!!\n",
    "text = \"\"\n",
    "for ind, row in train.iterrows():\n",
    "    text += row[\"Text_Headline\"] + \" \"\n",
    "text = text.strip()\n",
    "\n",
    "wordcloud = WordCloud(background_color='white', width=1000, height=600, max_font_size=100, max_words=50).generate(text)\n",
    "wordcloud.recolor(random_state=ind*312)\n",
    "plt.imshow(wordcloud)\n",
    "plt.axis(\"off\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [],
   "source": [
    "## now doing some vectorization method for transforming bour dataset!! for training data set"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "vectorizer = TfidfVectorizer(use_idf=True)\n",
    "\n",
    "train_v_Title = vectorizer.fit_transform(train['Text_Title'])\n",
    "test_v_Title = vectorizer.transform(test['Text_Title'])\n",
    "\n",
    "vectorizer_ = TfidfVectorizer()\n",
    "\n",
    "train_v_Headline = vectorizer_.fit_transform(train['Text_Headline'])\n",
    "test_v_Headline = vectorizer_.transform(test['Text_Headline'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(55932, 20)"
      ]
     },
     "execution_count": 82,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_v_Title.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(55932, 20)"
      ]
     },
     "execution_count": 84,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_v_Headline.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "# svd = TruncatedSVD(n_components=20)\n",
    "\n",
    "# train_v_Title = svd.fit_transform(train_v_Title)\n",
    "# test_v_Title = svd.transform(test_v_Title)\n",
    "\n",
    "# train_v_Headline = svd.fit_transform(train_v_Headline)\n",
    "# test_v_Headline = svd.transform(test_v_Headline)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [],
   "source": [
    "## now doing some vectorization method for transforming our dataset!! for testing data set"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now moving forward to core of our project to calculate the sentiment of our text data!!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [],
   "source": [
    "train['polarity_title'] = train['Title'].apply(lambda x: TextBlob(x).sentiment.polarity)\n",
    "test['polarity_title'] = test['Title'].apply(lambda x: TextBlob(x).sentiment.polarity)\n",
    "\n",
    "train['subjectivity_title'] = train['Title'].apply(lambda x: TextBlob(x).sentiment.subjectivity)\n",
    "test['subjectivity_title'] = test['Title'].apply(lambda x: TextBlob(x).sentiment.subjectivity)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0        0.00\n",
       "1        0.00\n",
       "2        0.00\n",
       "3        0.00\n",
       "4        0.00\n",
       "         ... \n",
       "55927    0.00\n",
       "55928    0.25\n",
       "55929   -0.70\n",
       "55930    0.00\n",
       "55931    0.00\n",
       "Name: polarity_title, Length: 55932, dtype: float64"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train['polarity_title']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [],
   "source": [
    "train['polarity_headline'] = train['Headline'].apply(lambda x: TextBlob(x).sentiment.polarity)\n",
    "test['polarity_headline'] = test['Headline'].apply(lambda x: TextBlob(x).sentiment.polarity)\n",
    "\n",
    "train['subjectivity_headline'] = train['Headline'].apply(lambda x: TextBlob(x).sentiment.subjectivity)\n",
    "test['subjectivity_headline'] = test['Headline'].apply(lambda x: TextBlob(x).sentiment.subjectivity)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0        0.000000\n",
       "1        0.100000\n",
       "2        0.000000\n",
       "3       -0.166667\n",
       "4        0.133333\n",
       "           ...   \n",
       "55927    0.000000\n",
       "55928    0.021429\n",
       "55929    0.088690\n",
       "55930    0.000000\n",
       "55931   -0.700000\n",
       "Name: polarity_headline, Length: 55932, dtype: float64"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train['polarity_headline']"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    " now encoding the colum,n topic and source by creating the dummies values!!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [],
   "source": [
    "encoder = LabelEncoder()\n",
    "\n",
    "train['Topic'] = encoder.fit_transform(train['Topic'])\n",
    "test['Topic'] = encoder.transform(test['Topic'])\n",
    "\n",
    "total = train['Source'].to_list() + test['Source'].to_list()\n",
    "total = encoder.fit_transform(total)\n",
    "train['Source'] = encoder.transform(train['Source'])\n",
    "test['Source'] = encoder.transform(test['Source'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0        4995\n",
       "1         518\n",
       "2         518\n",
       "3        3621\n",
       "4        4597\n",
       "         ... \n",
       "55927    5360\n",
       "55928    2500\n",
       "55929    2010\n",
       "55930    4185\n",
       "55931    5382\n",
       "Name: Source, Length: 55932, dtype: int32"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train['Source']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0         659\n",
       "1        1146\n",
       "2        1350\n",
       "3         524\n",
       "4        1556\n",
       "         ... \n",
       "37283    2767\n",
       "37284    4121\n",
       "37285    1132\n",
       "37286     169\n",
       "37287    1359\n",
       "Name: Source, Length: 37288, dtype: int32"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test['Source']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [],
   "source": [
    "# day type monday to sunday from datetime!!\n",
    "\n",
    "train_weekday = []\n",
    "test_weekday = []\n",
    "\n",
    "for i in train['PublishDate']:\n",
    "    train_weekday.append(datetime.datetime.strptime(i, \"%Y-%m-%d %H:%M:%S\").strftime(\"%A\"))\n",
    "    \n",
    "for i in test['PublishDate']:\n",
    "    test_weekday.append(datetime.datetime.strptime(i, \"%Y-%m-%d %H:%M:%S\").strftime(\"%A\"))\n",
    "\n",
    "train['weekday'] = train_weekday\n",
    "test['weekday'] = test_weekday\n",
    "\n",
    "\n",
    "# convert weekday to 1-7 of train data\n",
    "\n",
    "train['weekday'] = train['weekday'].map({'Monday': 0,\n",
    "                                        'Tuesday': 1,\n",
    "                                        'Wednesday': 2,\n",
    "                                        'Thursday': 3,\n",
    "                                        'Friday': 4,\n",
    "                                        'Saturday': 5,\n",
    "                                        'Sunday': 6})\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [],
   "source": [
    "# convert weekday to 1-7 of test data\n",
    "test['weekday'] = test['weekday'].map({'Monday': 0,\n",
    "                                        'Tuesday': 1,\n",
    "                                        'Wednesday': 2,\n",
    "                                        'Thursday': 3,\n",
    "                                        'Friday': 4,\n",
    "                                        'Saturday': 5,\n",
    "                                        'Sunday': 6})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [],
   "source": [
    "train[\"hour\"] = train[\"PublishDate\"].apply(lambda x: x.split()[1].split(':')[0])\n",
    "test[\"hour\"] = test[\"PublishDate\"].apply(lambda x: x.split()[1].split(':')[0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0, 0.5, 'SentimentTitle')"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "\n",
    "# plotting of the hour vs sentiminet title \n",
    "plt.scatter(train['hour'], train['SentimentTitle'], color= 'green')\n",
    "plt.xlabel('hour')\n",
    "plt.ylabel('SentimentTitle')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0, 0.5, 'SentimentHeadline')"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# distribution plot for hour vs sentimental headlines\n",
    "\n",
    "plt.scatter(train['hour'], train['SentimentHeadline'], color='red')\n",
    "plt.xlabel('hour')\n",
    "plt.ylabel('SentimentHeadline')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0, 0.5, 'SentimentTitle')"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# weekday distribution for weekdays and setimental Title!!\n",
    "\n",
    "plt.scatter(train['weekday'], train['SentimentTitle'])\n",
    "plt.xlabel('weekday')\n",
    "plt.ylabel('SentimentTitle')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0, 0.5, 'SentimentHeadline')"
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# weekday distribution for weekdays and setimental Headlines!!\n",
    "\n",
    "\n",
    "plt.scatter(train['weekday'], train['SentimentHeadline'], color= 'violet')\n",
    "plt.xlabel('weekday')\n",
    "plt.ylabel('SentimentHeadline')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0       -0.053300\n",
       "1       -0.156386\n",
       "2        0.139754\n",
       "3        0.026064\n",
       "4        0.141084\n",
       "           ...   \n",
       "55927   -0.055902\n",
       "55928    0.056110\n",
       "55929    0.114820\n",
       "55930   -0.028296\n",
       "55931    0.184444\n",
       "Name: SentimentHeadline, Length: 55932, dtype: float64"
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train['SentimentHeadline']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [],
   "source": [
    "# figuuring out the number of words in the title!!\n",
    "\n",
    "# Number of words in the Title \n",
    "train[\"num_words_title\"] = train[\"Text_Title\"].apply(lambda x: len(str(x).split()))\n",
    "test[\"num_words_title\"] = test[\"Text_Title\"].apply(lambda x: len(str(x).split()))\n",
    "\n",
    "# Number of unique words in the Title \n",
    "train[\"num_unique_words_title\"] = train[\"Text_Title\"].apply(lambda x: len(set(str(x).split())))\n",
    "test[\"num_unique_words_title\"] = test[\"Text_Title\"].apply(lambda x: len(set(str(x).split())))\n",
    "\n",
    "# Number of characters in the Title \n",
    "train[\"num_chars_title\"] = train[\"Text_Title\"].apply(lambda x: len(str(x)))\n",
    "test[\"num_chars_title\"] = test[\"Text_Title\"].apply(lambda x: len(str(x)))\n",
    "\n",
    "# Average length of the words in the Title \n",
    "train[\"mean_word_len_title\"] = train[\"Text_Title\"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))\n",
    "test[\"mean_word_len_title\"] = test[\"Text_Title\"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [],
   "source": [
    "# figuuring out the number of words in the title!!\n",
    "\n",
    "train[\"num_words_headline\"] = train[\"Text_Headline\"].apply(lambda x: len(str(x).split()))\n",
    "test[\"num_words_headline\"] = test[\"Text_Headline\"].apply(lambda x: len(str(x).split()))\n",
    "\n",
    "# Number of unique words in the Headline \n",
    "train[\"num_unique_words_headline\"] = train[\"Text_Headline\"].apply(lambda x: len(set(str(x).split())))\n",
    "test[\"num_unique_words_headline\"] = test[\"Text_Headline\"].apply(lambda x: len(set(str(x).split())))\n",
    "\n",
    "# Number of characters in the Headline \n",
    "train[\"num_chars_headline\"] = train[\"Text_Headline\"].apply(lambda x: len(str(x)))\n",
    "test[\"num_chars_headline\"] = test[\"Text_Headline\"].apply(lambda x: len(str(x)))\n",
    "\n",
    "# Average length of the words in the Headline \n",
    "train[\"mean_word_len_headline\"] = train[\"Text_Headline\"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))\n",
    "test[\"mean_word_len_headline\"] = test[\"Text_Headline\"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [],
   "source": [
    "## now standardization with scalar module!!\n",
    "\n",
    "scaler = StandardScaler()\n",
    "\n",
    "cols = ['Source', 'Topic', 'Facebook', 'GooglePlus', 'LinkedIn', 'num_words_title', 'num_unique_words_title', 'num_chars_title', 'mean_word_len_title',\n",
    "        'num_words_headline', 'num_unique_words_headline', 'num_chars_headline', 'mean_word_len_headline', 'hour', 'weekday']\n",
    "\n",
    "for col in cols:\n",
    "  train[col] = scaler.fit_transform(train[col].values.reshape(-1, 1))\n",
    "  test[col] = scaler.transform(test[col].values.reshape(-1, 1))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "# check the columns titles and headlines \n",
    "\n",
    "cols_title = ['Source', 'Topic', 'Facebook', 'GooglePlus', 'LinkedIn', 'num_words_title', 'num_unique_words_title', 'num_chars_title', 'mean_word_len_title', 'polarity_title', 'subjectivity_title', 'hour', 'weekday']\n",
    "train_X1 = train[cols_title]\n",
    "test_X1 = test[cols_title]\n",
    "\n",
    "cols_headline = ['Source', 'Topic', 'Facebook', 'GooglePlus', 'LinkedIn', 'num_words_headline', 'num_unique_words_headline', 'num_chars_headline', 'mean_word_len_headline', 'polarity_headline', 'subjectivity_headline', 'hour', 'weekday']\n",
    "train_X2 = train[cols_headline]\n",
    "test_X2 = test[cols_headline]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Source</th>\n",
       "      <th>Topic</th>\n",
       "      <th>Facebook</th>\n",
       "      <th>GooglePlus</th>\n",
       "      <th>LinkedIn</th>\n",
       "      <th>num_words_title</th>\n",
       "      <th>num_unique_words_title</th>\n",
       "      <th>num_chars_title</th>\n",
       "      <th>mean_word_len_title</th>\n",
       "      <th>polarity_title</th>\n",
       "      <th>subjectivity_title</th>\n",
       "      <th>hour</th>\n",
       "      <th>weekday</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>0</td>\n",
       "      <td>1.257700</td>\n",
       "      <td>0.841443</td>\n",
       "      <td>-0.184044</td>\n",
       "      <td>-0.262649</td>\n",
       "      <td>-0.199608</td>\n",
       "      <td>-0.259144</td>\n",
       "      <td>-0.304659</td>\n",
       "      <td>-0.572668</td>\n",
       "      <td>-0.612899</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>-1.699073</td>\n",
       "      <td>-0.795924</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1</td>\n",
       "      <td>-1.314091</td>\n",
       "      <td>-1.108773</td>\n",
       "      <td>-0.184044</td>\n",
       "      <td>-0.262649</td>\n",
       "      <td>-0.199608</td>\n",
       "      <td>-1.668699</td>\n",
       "      <td>-1.721681</td>\n",
       "      <td>-1.629548</td>\n",
       "      <td>0.274497</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>-1.699073</td>\n",
       "      <td>1.338369</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2</td>\n",
       "      <td>-1.314091</td>\n",
       "      <td>-1.108773</td>\n",
       "      <td>-0.184044</td>\n",
       "      <td>-0.262649</td>\n",
       "      <td>-0.199608</td>\n",
       "      <td>-1.198848</td>\n",
       "      <td>-1.249340</td>\n",
       "      <td>-1.035053</td>\n",
       "      <td>0.328828</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>-1.699073</td>\n",
       "      <td>1.338369</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3</td>\n",
       "      <td>0.468412</td>\n",
       "      <td>-1.108773</td>\n",
       "      <td>-0.184044</td>\n",
       "      <td>-0.262649</td>\n",
       "      <td>-0.199608</td>\n",
       "      <td>-1.668699</td>\n",
       "      <td>-1.249340</td>\n",
       "      <td>-2.224043</td>\n",
       "      <td>-1.436911</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>-1.699073</td>\n",
       "      <td>1.871942</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4</td>\n",
       "      <td>1.029071</td>\n",
       "      <td>-1.108773</td>\n",
       "      <td>-0.184044</td>\n",
       "      <td>-0.262649</td>\n",
       "      <td>-0.199608</td>\n",
       "      <td>1.150411</td>\n",
       "      <td>1.112364</td>\n",
       "      <td>1.012652</td>\n",
       "      <td>-0.295972</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>-1.699073</td>\n",
       "      <td>1.871942</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     Source     Topic  Facebook  GooglePlus  LinkedIn  num_words_title  \\\n",
       "0  1.257700  0.841443 -0.184044   -0.262649 -0.199608        -0.259144   \n",
       "1 -1.314091 -1.108773 -0.184044   -0.262649 -0.199608        -1.668699   \n",
       "2 -1.314091 -1.108773 -0.184044   -0.262649 -0.199608        -1.198848   \n",
       "3  0.468412 -1.108773 -0.184044   -0.262649 -0.199608        -1.668699   \n",
       "4  1.029071 -1.108773 -0.184044   -0.262649 -0.199608         1.150411   \n",
       "\n",
       "   num_unique_words_title  num_chars_title  mean_word_len_title  \\\n",
       "0               -0.304659        -0.572668            -0.612899   \n",
       "1               -1.721681        -1.629548             0.274497   \n",
       "2               -1.249340        -1.035053             0.328828   \n",
       "3               -1.249340        -2.224043            -1.436911   \n",
       "4                1.112364         1.012652            -0.295972   \n",
       "\n",
       "   polarity_title  subjectivity_title      hour   weekday  \n",
       "0             0.0                 0.0 -1.699073 -0.795924  \n",
       "1             0.0                 0.0 -1.699073  1.338369  \n",
       "2             0.0                 0.0 -1.699073  1.338369  \n",
       "3             0.0                 0.0 -1.699073  1.871942  \n",
       "4             0.0                 0.0 -1.699073  1.871942  "
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_X1.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Source</th>\n",
       "      <th>Topic</th>\n",
       "      <th>Facebook</th>\n",
       "      <th>GooglePlus</th>\n",
       "      <th>LinkedIn</th>\n",
       "      <th>num_words_headline</th>\n",
       "      <th>num_unique_words_headline</th>\n",
       "      <th>num_chars_headline</th>\n",
       "      <th>mean_word_len_headline</th>\n",
       "      <th>polarity_headline</th>\n",
       "      <th>subjectivity_headline</th>\n",
       "      <th>hour</th>\n",
       "      <th>weekday</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>0</td>\n",
       "      <td>1.257700</td>\n",
       "      <td>0.841443</td>\n",
       "      <td>-0.184044</td>\n",
       "      <td>-0.262649</td>\n",
       "      <td>-0.199608</td>\n",
       "      <td>-0.285943</td>\n",
       "      <td>-0.526753</td>\n",
       "      <td>-0.490533</td>\n",
       "      <td>-1.058531</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>-1.699073</td>\n",
       "      <td>-0.795924</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1</td>\n",
       "      <td>-1.314091</td>\n",
       "      <td>-1.108773</td>\n",
       "      <td>-0.184044</td>\n",
       "      <td>-0.262649</td>\n",
       "      <td>-0.199608</td>\n",
       "      <td>-0.285943</td>\n",
       "      <td>-0.232503</td>\n",
       "      <td>-0.317671</td>\n",
       "      <td>-0.175891</td>\n",
       "      <td>0.100000</td>\n",
       "      <td>0.200000</td>\n",
       "      <td>-1.699073</td>\n",
       "      <td>1.338369</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2</td>\n",
       "      <td>-1.314091</td>\n",
       "      <td>-1.108773</td>\n",
       "      <td>-0.184044</td>\n",
       "      <td>-0.262649</td>\n",
       "      <td>-0.199608</td>\n",
       "      <td>-0.413305</td>\n",
       "      <td>-0.673878</td>\n",
       "      <td>-0.300385</td>\n",
       "      <td>0.618485</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.041667</td>\n",
       "      <td>-1.699073</td>\n",
       "      <td>1.338369</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3</td>\n",
       "      <td>0.468412</td>\n",
       "      <td>-1.108773</td>\n",
       "      <td>-0.184044</td>\n",
       "      <td>-0.262649</td>\n",
       "      <td>-0.199608</td>\n",
       "      <td>0.096143</td>\n",
       "      <td>0.061747</td>\n",
       "      <td>0.304632</td>\n",
       "      <td>0.843558</td>\n",
       "      <td>-0.166667</td>\n",
       "      <td>0.166667</td>\n",
       "      <td>-1.699073</td>\n",
       "      <td>1.871942</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4</td>\n",
       "      <td>1.029071</td>\n",
       "      <td>-1.108773</td>\n",
       "      <td>-0.184044</td>\n",
       "      <td>-0.262649</td>\n",
       "      <td>-0.199608</td>\n",
       "      <td>0.223505</td>\n",
       "      <td>0.208872</td>\n",
       "      <td>0.391063</td>\n",
       "      <td>0.618485</td>\n",
       "      <td>0.133333</td>\n",
       "      <td>0.380556</td>\n",
       "      <td>-1.699073</td>\n",
       "      <td>1.871942</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     Source     Topic  Facebook  GooglePlus  LinkedIn  num_words_headline  \\\n",
       "0  1.257700  0.841443 -0.184044   -0.262649 -0.199608           -0.285943   \n",
       "1 -1.314091 -1.108773 -0.184044   -0.262649 -0.199608           -0.285943   \n",
       "2 -1.314091 -1.108773 -0.184044   -0.262649 -0.199608           -0.413305   \n",
       "3  0.468412 -1.108773 -0.184044   -0.262649 -0.199608            0.096143   \n",
       "4  1.029071 -1.108773 -0.184044   -0.262649 -0.199608            0.223505   \n",
       "\n",
       "   num_unique_words_headline  num_chars_headline  mean_word_len_headline  \\\n",
       "0                  -0.526753           -0.490533               -1.058531   \n",
       "1                  -0.232503           -0.317671               -0.175891   \n",
       "2                  -0.673878           -0.300385                0.618485   \n",
       "3                   0.061747            0.304632                0.843558   \n",
       "4                   0.208872            0.391063                0.618485   \n",
       "\n",
       "   polarity_headline  subjectivity_headline      hour   weekday  \n",
       "0           0.000000               0.000000 -1.699073 -0.795924  \n",
       "1           0.100000               0.200000 -1.699073  1.338369  \n",
       "2           0.000000               0.041667 -1.699073  1.338369  \n",
       "3          -0.166667               0.166667 -1.699073  1.871942  \n",
       "4           0.133333               0.380556 -1.699073  1.871942  "
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_X2.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Source</th>\n",
       "      <th>Topic</th>\n",
       "      <th>Facebook</th>\n",
       "      <th>GooglePlus</th>\n",
       "      <th>LinkedIn</th>\n",
       "      <th>num_words_title</th>\n",
       "      <th>num_unique_words_title</th>\n",
       "      <th>num_chars_title</th>\n",
       "      <th>mean_word_len_title</th>\n",
       "      <th>polarity_title</th>\n",
       "      <th>subjectivity_title</th>\n",
       "      <th>hour</th>\n",
       "      <th>weekday</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>0</td>\n",
       "      <td>-1.233094</td>\n",
       "      <td>-1.108773</td>\n",
       "      <td>-0.182661</td>\n",
       "      <td>-0.215339</td>\n",
       "      <td>-0.173516</td>\n",
       "      <td>-0.259144</td>\n",
       "      <td>-0.304659</td>\n",
       "      <td>-0.440558</td>\n",
       "      <td>-0.359357</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.00</td>\n",
       "      <td>-1.555397</td>\n",
       "      <td>-0.795924</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1</td>\n",
       "      <td>-0.953339</td>\n",
       "      <td>-0.133665</td>\n",
       "      <td>-0.015286</td>\n",
       "      <td>-0.120718</td>\n",
       "      <td>-0.016962</td>\n",
       "      <td>0.210707</td>\n",
       "      <td>0.167682</td>\n",
       "      <td>0.352102</td>\n",
       "      <td>0.084341</td>\n",
       "      <td>-0.1</td>\n",
       "      <td>0.35</td>\n",
       "      <td>-1.555397</td>\n",
       "      <td>-0.795924</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2</td>\n",
       "      <td>-0.836153</td>\n",
       "      <td>-0.133665</td>\n",
       "      <td>-0.166062</td>\n",
       "      <td>-0.168028</td>\n",
       "      <td>-0.186562</td>\n",
       "      <td>0.210707</td>\n",
       "      <td>0.167682</td>\n",
       "      <td>0.418157</td>\n",
       "      <td>0.312529</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.00</td>\n",
       "      <td>-1.555397</td>\n",
       "      <td>-0.795924</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3</td>\n",
       "      <td>-1.310644</td>\n",
       "      <td>-1.108773</td>\n",
       "      <td>-0.182661</td>\n",
       "      <td>-0.215339</td>\n",
       "      <td>-0.147424</td>\n",
       "      <td>2.090114</td>\n",
       "      <td>2.057046</td>\n",
       "      <td>1.078707</td>\n",
       "      <td>-1.382580</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.25</td>\n",
       "      <td>-1.411721</td>\n",
       "      <td>-0.795924</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4</td>\n",
       "      <td>-0.717817</td>\n",
       "      <td>-1.108773</td>\n",
       "      <td>-0.178511</td>\n",
       "      <td>-0.215339</td>\n",
       "      <td>-0.186562</td>\n",
       "      <td>-0.259144</td>\n",
       "      <td>-0.304659</td>\n",
       "      <td>-0.506613</td>\n",
       "      <td>-0.486128</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.25</td>\n",
       "      <td>-1.411721</td>\n",
       "      <td>-0.795924</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     Source     Topic  Facebook  GooglePlus  LinkedIn  num_words_title  \\\n",
       "0 -1.233094 -1.108773 -0.182661   -0.215339 -0.173516        -0.259144   \n",
       "1 -0.953339 -0.133665 -0.015286   -0.120718 -0.016962         0.210707   \n",
       "2 -0.836153 -0.133665 -0.166062   -0.168028 -0.186562         0.210707   \n",
       "3 -1.310644 -1.108773 -0.182661   -0.215339 -0.147424         2.090114   \n",
       "4 -0.717817 -1.108773 -0.178511   -0.215339 -0.186562        -0.259144   \n",
       "\n",
       "   num_unique_words_title  num_chars_title  mean_word_len_title  \\\n",
       "0               -0.304659        -0.440558            -0.359357   \n",
       "1                0.167682         0.352102             0.084341   \n",
       "2                0.167682         0.418157             0.312529   \n",
       "3                2.057046         1.078707            -1.382580   \n",
       "4               -0.304659        -0.506613            -0.486128   \n",
       "\n",
       "   polarity_title  subjectivity_title      hour   weekday  \n",
       "0             0.0                0.00 -1.555397 -0.795924  \n",
       "1            -0.1                0.35 -1.555397 -0.795924  \n",
       "2             0.0                0.00 -1.555397 -0.795924  \n",
       "3             0.0                0.25 -1.411721 -0.795924  \n",
       "4             0.0                0.25 -1.411721 -0.795924  "
      ]
     },
     "execution_count": 49,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_X1.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Source</th>\n",
       "      <th>Topic</th>\n",
       "      <th>Facebook</th>\n",
       "      <th>GooglePlus</th>\n",
       "      <th>LinkedIn</th>\n",
       "      <th>num_words_headline</th>\n",
       "      <th>num_unique_words_headline</th>\n",
       "      <th>num_chars_headline</th>\n",
       "      <th>mean_word_len_headline</th>\n",
       "      <th>polarity_headline</th>\n",
       "      <th>subjectivity_headline</th>\n",
       "      <th>hour</th>\n",
       "      <th>weekday</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>0</td>\n",
       "      <td>-1.233094</td>\n",
       "      <td>-1.108773</td>\n",
       "      <td>-0.182661</td>\n",
       "      <td>-0.215339</td>\n",
       "      <td>-0.173516</td>\n",
       "      <td>-0.668030</td>\n",
       "      <td>-0.526753</td>\n",
       "      <td>-0.490533</td>\n",
       "      <td>1.047196</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>-1.555397</td>\n",
       "      <td>-0.795924</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1</td>\n",
       "      <td>-0.953339</td>\n",
       "      <td>-0.133665</td>\n",
       "      <td>-0.015286</td>\n",
       "      <td>-0.120718</td>\n",
       "      <td>-0.016962</td>\n",
       "      <td>-0.413305</td>\n",
       "      <td>-0.379628</td>\n",
       "      <td>-0.404102</td>\n",
       "      <td>-0.037978</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.250000</td>\n",
       "      <td>-1.555397</td>\n",
       "      <td>-0.795924</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2</td>\n",
       "      <td>-0.836153</td>\n",
       "      <td>-0.133665</td>\n",
       "      <td>-0.166062</td>\n",
       "      <td>-0.168028</td>\n",
       "      <td>-0.186562</td>\n",
       "      <td>2.133938</td>\n",
       "      <td>1.974371</td>\n",
       "      <td>2.517265</td>\n",
       "      <td>0.743526</td>\n",
       "      <td>-0.157407</td>\n",
       "      <td>0.251852</td>\n",
       "      <td>-1.555397</td>\n",
       "      <td>-0.795924</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3</td>\n",
       "      <td>-1.310644</td>\n",
       "      <td>-1.108773</td>\n",
       "      <td>-0.182661</td>\n",
       "      <td>-0.215339</td>\n",
       "      <td>-0.147424</td>\n",
       "      <td>0.096143</td>\n",
       "      <td>0.208872</td>\n",
       "      <td>-0.110237</td>\n",
       "      <td>-0.957027</td>\n",
       "      <td>0.087500</td>\n",
       "      <td>0.272917</td>\n",
       "      <td>-1.411721</td>\n",
       "      <td>-0.795924</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4</td>\n",
       "      <td>-0.717817</td>\n",
       "      <td>-1.108773</td>\n",
       "      <td>-0.178511</td>\n",
       "      <td>-0.215339</td>\n",
       "      <td>-0.186562</td>\n",
       "      <td>-0.413305</td>\n",
       "      <td>-0.379628</td>\n",
       "      <td>-0.542392</td>\n",
       "      <td>-0.694442</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>-1.411721</td>\n",
       "      <td>-0.795924</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     Source     Topic  Facebook  GooglePlus  LinkedIn  num_words_headline  \\\n",
       "0 -1.233094 -1.108773 -0.182661   -0.215339 -0.173516           -0.668030   \n",
       "1 -0.953339 -0.133665 -0.015286   -0.120718 -0.016962           -0.413305   \n",
       "2 -0.836153 -0.133665 -0.166062   -0.168028 -0.186562            2.133938   \n",
       "3 -1.310644 -1.108773 -0.182661   -0.215339 -0.147424            0.096143   \n",
       "4 -0.717817 -1.108773 -0.178511   -0.215339 -0.186562           -0.413305   \n",
       "\n",
       "   num_unique_words_headline  num_chars_headline  mean_word_len_headline  \\\n",
       "0                  -0.526753           -0.490533                1.047196   \n",
       "1                  -0.379628           -0.404102               -0.037978   \n",
       "2                   1.974371            2.517265                0.743526   \n",
       "3                   0.208872           -0.110237               -0.957027   \n",
       "4                  -0.379628           -0.542392               -0.694442   \n",
       "\n",
       "   polarity_headline  subjectivity_headline      hour   weekday  \n",
       "0           0.000000               0.000000 -1.555397 -0.795924  \n",
       "1           0.000000               0.250000 -1.555397 -0.795924  \n",
       "2          -0.157407               0.251852 -1.555397 -0.795924  \n",
       "3           0.087500               0.272917 -1.411721 -0.795924  \n",
       "4           0.000000               0.000000 -1.411721 -0.795924  "
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_X2.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['IDLink', 'Title', 'Headline', 'Source', 'Topic', 'PublishDate',\n",
       "       'Facebook', 'GooglePlus', 'LinkedIn', 'SentimentTitle',\n",
       "       'SentimentHeadline', 'Text_Title', 'Text_Headline', 'polarity_title',\n",
       "       'subjectivity_title', 'polarity_headline', 'subjectivity_headline',\n",
       "       'weekday', 'hour', 'num_words_title', 'num_unique_words_title',\n",
       "       'num_chars_title', 'mean_word_len_title', 'num_words_headline',\n",
       "       'num_unique_words_headline', 'num_chars_headline',\n",
       "       'mean_word_len_headline'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 56,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(55932, 13) train X1\n",
      "(37288, 13) test X1\n"
     ]
    }
   ],
   "source": [
    "# checking the shape of the data\n",
    "\n",
    "print(train_X1.shape, 'train X1')\n",
    "print(test_X1.shape, 'test X1')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(55932, 13) train X2\n",
      "(37288, 13) test X2\n"
     ]
    }
   ],
   "source": [
    "print(train_X2.shape, 'train X2')\n",
    "print(test_X2.shape, 'test X2')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(55932, 20)\n",
      "(37288, 20)\n"
     ]
    }
   ],
   "source": [
    "\n",
    "print(np.shape(train_v_Title))\n",
    "print(np.shape(test_v_Title))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(55932, 20)\n",
      "(37288, 20)\n"
     ]
    }
   ],
   "source": [
    "print(np.shape(train_v_Headline))\n",
    "print(np.shape(test_v_Headline))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(55932, 20)\n",
      "(37288, 20)\n"
     ]
    }
   ],
   "source": [
    "print(np.shape(train_v_Title))\n",
    "print(np.shape(test_v_Title))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(55932, 20)\n",
      "(37288, 20)\n"
     ]
    }
   ],
   "source": [
    "print(np.shape(train_v_Headline))\n",
    "print(np.shape(test_v_Headline))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "reducing the dimesnion for better performance of the model using SVD!!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 0.16775825 -0.11412548 -0.04822087 ... -0.00986373 -0.01220966\n",
      "  -0.02261364]\n",
      " [ 0.078058    0.07474753  0.11667698 ...  0.04775589 -0.05251913\n",
      "  -0.09417374]\n",
      " [ 0.0533445   0.05564995  0.10977539 ...  0.0646559  -0.00551403\n",
      "  -0.04585838]\n",
      " ...\n",
      " [ 0.1630509  -0.09041928 -0.03639916 ...  0.02606334 -0.05286184\n",
      "  -0.00654206]\n",
      " [ 0.07765037  0.18697134 -0.14515641 ... -0.0324557  -0.00185224\n",
      "  -0.01104474]\n",
      " [ 0.1170965   0.04986726  0.13595587 ...  0.07324378 -0.0077704\n",
      "  -0.02188705]] train headlines\n",
      "[[ 0.08542381  0.00946836  0.06325832 ... -0.02996335 -0.02102944\n",
      "   0.01318694]\n",
      " [ 0.09013356  0.16647582 -0.12805879 ... -0.00569838 -0.02772593\n",
      "  -0.00278818]\n",
      " [ 0.04638863  0.10680215 -0.08848141 ... -0.0041639  -0.01038824\n",
      "  -0.00113923]\n",
      " ...\n",
      " [ 0.0451431   0.01544674  0.01889963 ... -0.00841987 -0.01385428\n",
      "   0.01302112]\n",
      " [ 0.03321775  0.01313172  0.00825601 ...  0.01274547  0.02344735\n",
      "  -0.03616345]\n",
      " [ 0.07408856  0.13338602 -0.10063309 ... -0.02347683  0.03425783\n",
      "  -0.00212084]] test  headlines\n"
     ]
    }
   ],
   "source": [
    "print(train_v_Headline, 'train headlines')\n",
    "print(test_v_Headline, 'test  headlines')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "train_X_Title = hstack([train_v_Title, csr_matrix(train_X1.values)])\n",
    "test_X_Title = hstack([test_v_Title, csr_matrix(test_X1.values)])\n",
    "y1 = train['SentimentTitle']\n",
    "\n",
    "train_X_Headline = hstack([train_v_Headline, csr_matrix(train_X2.values)])\n",
    "test_X_Headline = hstack([test_v_Headline, csr_matrix(test_X2.values)])\n",
    "y2 = train['SentimentHeadline']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(55932, 33) train x title\n",
      "(37288, 33) test x title\n",
      "(55932, 33) train x Headline\n",
      "(37288, 33) test x Headline\n"
     ]
    }
   ],
   "source": [
    "print(train_X_Title.shape,'train x title')\n",
    "print(test_X_Title.shape,'test x title')\n",
    "print(train_X_Headline.shape,'train x Headline')\n",
    "print(test_X_Headline.shape,'test x Headline')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(55932, 33)"
      ]
     },
     "execution_count": 68,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.shape(train_X_Title)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_X_Title = hstack([train_v_Title, csr_matrix(train_X1.values)])\n",
    "test_X_Title = hstack([test_v_Title, csr_matrix(test_X1.values)])\n",
    "y1 = train['SentimentTitle']\n",
    "\n",
    "train_X_Headline = hstack([train_v_Headline, csr_matrix(train_X2.values)])\n",
    "test_X_Headline = hstack([test_v_Headline, csr_matrix(test_X2.values)])\n",
    "y2 = train['SentimentHeadline']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(55932, 33)"
      ]
     },
     "execution_count": 70,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.shape(train_X_Title)\n",
    "# np.shape(test_v_Title)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Selecting Machine learning model for best MAE results!!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "MAE: 0.9055280809735686\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\ProgramData\\Anaconda3\\lib\\site-packages\\sklearn\\svm\\base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.\n",
      "  \"the number of iterations.\", ConvergenceWarning)\n"
     ]
    }
   ],
   "source": [
    "# 1) select support vector m,achine model now with c= 0.2!\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(train_X_Title, y1, test_size=0.20, random_state=42)\n",
    "\n",
    "clf1 = LinearSVR(C=0.2)\n",
    "clf1.fit(X_train, y_train)\n",
    "\n",
    "y_pred1 = clf1.predict(X_test)\n",
    "mae1 = mean_absolute_error(y_pred1, y_test)\n",
    "print('MAE:', 1 - mae1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "MAE: 0.8946623612719957\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\ProgramData\\Anaconda3\\lib\\site-packages\\sklearn\\svm\\base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.\n",
      "  \"the number of iterations.\", ConvergenceWarning)\n"
     ]
    }
   ],
   "source": [
    "# 1) select support vector m,achine model now with c= 3!\n",
    "\n",
    "\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(train_X_Headline, y2, test_size=0.20, random_state=42)\n",
    "\n",
    "clf2 = LinearSVR(C=0.3)\n",
    "clf2.fit(X_train, y_train)\n",
    "\n",
    "y_pred2 = clf2.predict(X_test)\n",
    "mae2 = mean_absolute_error(y_pred2, y_test)\n",
    "print('MAE:', 1 - mae2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "# X_train, X_test, y_train, y_test = train_test_split(train_X_Headline, y2, test_size=0.20, random_state=42)\n",
    "\n",
    "# # clf3 = LinearSVR(C=0.9)\n",
    "# # clf3.fit(X_train, y_train)\n",
    "\n",
    "# # y_pred3 = clf3.predict(X_test)\n",
    "# # mae3 = mean_absolute_error(y_pred3, y_test)\n",
    "# # print('MAE:', 1 - mae3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "metadata": {},
   "outputs": [],
   "source": [
    "## so our result with c= 0.9 in linear support vector regression is highest. so that we will use c= 0.9!!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## now checking the MAE( mean absolute error)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Leaderborad score= 0.8990086491526249\n"
     ]
    }
   ],
   "source": [
    "print('Leaderborad score=', max(0, 1 - ((0.4 * mae1) + (0.6 * mae2))))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now make prediction with test data!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "metadata": {},
   "outputs": [],
   "source": [
    "title = clf1.predict(test_X_Title)\n",
    "headline = clf2.predict(test_X_Headline)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.DataFrame()\n",
    "test_id = test['IDLink']\n",
    "df['IDLink'] = test_id\n",
    "df['SentimentTitle'] = title\n",
    "df['SentimentHeadline'] = headline\n",
    "df.to_csv('sample_submissions.csv', index = False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(11187,)"
      ]
     },
     "execution_count": 79,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_pred1.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "####  conclusion\n",
    "I have created the news sentiment analysis of the dataset provided using different metrics, and converted the sentiment title and headline into poloarity of the sentiments.. the more posive a setiment is the more it's positive and neagtive value of polarity is negative sentiments!!\n",
    "\n",
    "After analysing and using linear SVM modelling we reduced the MAE to 89.9 % with c= 0.2\n",
    "\n",
    "This model can be trained with other different news sentiment analysis for further use aswelll!!\n",
    "\n",
    "or can be transformed easily to use video and comments sentiment analysis of any ecommerce websites aswell!!\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
