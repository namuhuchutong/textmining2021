# ----------------------------------
#   
#   2021 TextMining Midterm
# 
#   Dataset : https://github.com/kocohub/korean-hate-speech
#   system("git clone https://github.com/kocohub/korean-hate-speech")
#   
#   조건1. 가장 자주 사용된 단어 추출 및 빈도 그래프 만들기
#   조건2. 오즈비 또는 TF-IDF 활용하여 분석하기(한 텍스트를 일정한 기준으로 2개 이상의 텍스트로 나누고 진행해도 무방함)
#   조건3. 감정사전을 적용하여, 텍스트의 감정 경향을 분석하기
#   조건4. 감정사전 수정하여 적용하고, 수정전과 비교분석하기
#   조건5. 2가지 이상의 네트워크 그래프를 그리고 비교분석하기
#   추가점수조건: github 로 버전관리 진행하고, 그 과정을 증빙하기(캡쳐 활용)
#   
# ----------------------------------

library(dplyr)
library(tidyr)
library(readr)
library(widyr)
library(textclean)
library(tidytext)
library(tidyselect)
library(tidygraph)
library(KoNLP)
library(stringr)
library(ggplot2)
library(ggraph)
#useNIADic()
useSejongDic()
dict <- read_csv("knu_sentiment_lexicon.csv")

raw_data <- read.delim("korean-hate-speech/labeled/train.tsv")
sum(is.na(raw_data))

hate_speeches <- raw_data %>% as_tibble()
hate_speeches <- hate_speeches %>% mutate(comments = str_replace_all(comments, "[^가-힣]", " "), comments = str_squish(comments))

hates <- hate_speeches %>% filter(str_detect(hate, "hate"))
offensives <- hate_speeches %>% filter(str_detect(hate, "offensive"))
nones <- hate_speeches %>% filter(str_detect(hate, "none"))

hates_tokens <- hates %>% unnest_tokens(input = comments, output = word, token = extractNoun)
freq_hates <- hates_tokens %>% count(bias, word) %>% filter(str_count(word) > 1)

offensives_tokens <- offensives %>% unnest_tokens(input = comments, output = word, token = extractNoun)
freq_offensives <- offensives_tokens %>% count(bias, word) %>% filter(str_count(word) > 1)

nones_tokens <- nones %>% unnest_tokens(input = comments, output = word, token = extractNoun)
freq_nones <- nones_tokens %>% count(bias, word) %>% filter(str_count(word) > 1)

hates_top10 <- freq_hates %>% group_by(bias) %>% slice_max(n, n=20)
offensives_top10 <- freq_offensives %>% group_by(bias) %>% slice_max(n, n=20)
nones_top10 <- freq_nones %>% group_by(bias) %>% slice_max(n, n = 20)


ggplot(hates_top10, aes(x = reorder(word, n), y = n, fill = bias)) + geom_col() + coord_flip() + facet_wrap(~bias, scales = "free_y") + theme(text=element_text(family ="AppleGothic"))
ggplot(offensives_top10, aes(x = reorder(word, n), y = n, fill = bias)) + geom_col() + coord_flip() + facet_wrap(~bias, scales = "free_y") + theme(text=element_text(family ="AppleGothic"))
ggplot(nones_top10, aes(x = reorder(word, n), y = n, fill = bias)) + geom_col() + coord_flip() + facet_wrap(~bias, scales = "free_y") + theme(text=element_text(family ="AppleGothic"))


hates_tfidf <- freq_hates %>% bind_tf_idf(term = word, document = bias, n = n) %>% arrange(-tf_idf)
hates_tfidf10 <- hates_tfidf %>% group_by(bias) %>% slice_max(tf_idf, n=20, with_ties = F)
hates_tfidf10$bias <- factor(hates_tfidf10$bias, levels = c("gender", "others", "none"))

offensives_tfidf <- freq_offensives %>% bind_tf_idf(term = word, document = bias, n = n) %>% arrange(-tf_idf)
offensives_tfidf10 <- offensives_tfidf %>% group_by(bias) %>% slice_max(tf_idf, n=20, with_ties = F)
offensives_tfidf10$bias <- factor(offensives_tfidf10$bias, levels = c("gender", "others", "none"))

nones_tfidf <- freq_nones %>% bind_tf_idf(term = word, document = bias, n = n) %>% arrange(-tf_idf)
nones_tfidf10 <- nones_tfidf %>% group_by(bias) %>% slice_max(tf_idf, n=20, with_ties = F)
nones_tfidf10$bias <- factor(nones_tfidf10$bias, levels = c("gender", "others", "none"))

ggplot(hates_tfidf10, aes(x=reorder_within(word, tf_idf, bias), y=tf_idf, fill=bias)) + geom_col(show.legend =  F) + coord_flip() + facet_wrap(~ bias, scales = "free", ncol = 2) + scale_x_reordered() + labs(x = NULL) + theme(text = element_text(family = "AppleGothic"))
ggplot(offensives_tfidf10, aes(x=reorder_within(word, tf_idf, bias), y=tf_idf, fill=bias)) + geom_col(show.legend =  F) + coord_flip() + facet_wrap(~ bias, scales = "free", ncol = 2) + scale_x_reordered() + labs(x = NULL) + theme(text = element_text(family = "AppleGothic"))
ggplot(nones_tfidf10, aes(x=reorder_within(word, tf_idf, bias), y=tf_idf, fill=bias)) + geom_col(show.legend =  F) + coord_flip() + facet_wrap(~ bias, scales = "free", ncol = 2) + scale_x_reordered() + labs(x = NULL) + theme(text = element_text(family = "AppleGothic"))

newwords <- tibble(word = c("한남", "꽃뱀", "쿵쾅이", "전라도", "쪽바리", "개티즌", "기레기", "한녀", "개소리", "개돼지"), polarity = -3)
new_dict <- bind_rows(dict, newwords)

raw_comments <- read.delim("korean-hate-speech/test.no_label.tsv")
new_comments <- raw_comments %>% mutate(id = row_number(), comments = str_replace_all(comments, "[^가-힣]", " "), comments = str_squish(comments))

word_new_comments <- new_comments %>% unnest_tokens(input = comments, output = word, token = "words", drop=F)
word_new_comments1 <- word_new_comments %>% left_join(new_dict, by="word") %>% mutate(polarity = ifelse(is.na(polarity), 0, polarity))
word_new_comments2 <- word_new_comments %>% left_join(dict, by="word") %>% mutate(polarity = ifelse(is.na(polarity), 0, polarity))

word_new_comments1_s <- word_new_comments1 %>% group_by(id, comments) %>% summarise(score = sum(polarity)) %>% ungroup()
word_new_comments2_s <- word_new_comments2 %>% group_by(id, comments) %>% summarise(score = sum(polarity)) %>% ungroup()

word_new_comments1_s <- word_new_comments1_s %>% mutate(sentiment = ifelse(score >= 1, "pos", ifelse(score <= -1, "neg", "neu")))
word_new_comments2_s <- word_new_comments2_s %>% mutate(sentiment = ifelse(score >= 1, "pos", ifelse(score <= -1, "neg", "neu")))


word_new_comments1_sf <- word_new_comments1_s %>% count(sentiment) %>% mutate(ratio = n/sum(n)*100)
word_new_comments2_sf <- word_new_comments2_s %>% count(sentiment) %>% mutate(ratio = n/sum(n)*100)

ggplot(word_new_comments1_sf, aes(x = sentiment, y = n, fill = sentiment)) + geom_col() + geom_text(aes(label = n), vjust = -0.3) + scale_x_discrete(limits = c("pos", "neu", "neg"))
ggplot(word_new_comments2_sf, aes(x = sentiment, y = n, fill = sentiment)) + geom_col() + geom_text(aes(label = n), vjust = -0.3) + scale_x_discrete(limits = c("pos", "neu", "neg"))

word_new_comments1_sf$dummy <- 0
word_new_comments2_sf$dummy <- 0

ggplot(word_new_comments1_sf, aes(x = dummy, y = ratio, fill = sentiment)) + geom_col() + geom_text(aes(label = paste0(round(ratio, 1), "%")), position = position_stack(vjust = 0.5)) + theme(axis.title = element_blank(), axis.text.x = element_blank(), axis.ticks.x = element_blank())
ggplot(word_new_comments2_sf, aes(x = dummy, y = ratio, fill = sentiment)) + geom_col() + geom_text(aes(label = paste0(round(ratio, 1), "%")), position = position_stack(vjust = 0.5)) + theme(axis.title = element_blank(), axis.text.x = element_blank(), axis.ticks.x = element_blank())

#raw_unlabeled_data <- readLines("korean-hate-speech/unlabeled/unlabeled_comments_5.txt")
#unlabeled_data <- raw_comments %>% as_tibble()
unlabeled_data <- hate_speeches %>% select(comments) %>% mutate( comments = str_replace_all(comments, "[^가-힣]", " "), value = str_squish(comments), id = row_number())

unlabeled_data_pos <- unlabeled_data %>% unnest_tokens(input = comments, output = word, token = SimplePos22, drop = F)
unlabeled_comments <- unlabeled_data_pos %>% separate_rows(word, sep="[+]") %>% filter(str_detect(word, "/n|/pv|/pa")) %>% mutate(word = ifelse( str_detect(word, "/pv|/pa"), str_replace(word, "/.*$", "다"), str_remove(word, "/.*$"))) %>% filter(str_count(word) >= 2) %>% arrange(id)

pair <- unlabeled_comments %>% pairwise_count(item = word, feature = id, sort = T)
graph_unlabeled_comments <- pair %>% filter(n >= 18) %>% as_tbl_graph()

ggraph(graph_unlabeled_comments, layout = "fr") + geom_edge_link(color = "gray50", alpha= 0.3) + geom_node_point(color = "lightcoral", size = 5) + geom_node_text(aes(label = name), repel = T, size = 5, family="AppleGothic") + theme_graph() 
set.seed(1234)
graph_unlabeled_comments <- pair %>% filter(n >= 18) %>% as_tbl_graph(directed = F) %>% mutate(centrality = centrality_degree(), group = as.factor(group_infomap()))
set.seed(1234)
ggraph(graph_unlabeled_comments, layout = "fr") + geom_edge_link(color = "gray50", alpha = 0.5) + geom_node_point(aes(size = centrality, color = group) , show.legend =  F) + scale_size(range = c(5, 15)) + geom_node_text(aes(label=name), repel= T, size = 5, family="AppleGothic") + theme_graph()
graph_unlabeled_comments %>% arrange(-centrality)

word_cors <- unlabeled_comments %>% add_count(word) %>% filter(n >= 18) %>% pairwise_cor(item = word, feature = id, sort = T)
graph_cors <- word_cors %>% filter(correlation >= 0.1) %>% as_tbl_graph(directed = F) %>% mutate(centrality = centrality_degree(), group = as.factor(group_infomap()))

ggraph(graph_cors, layout = "fr") + geom_edge_link(color = "gray50", aes(edge_alpha = correlation, edge_width = correlation), show.legend =  F) + scale_edge_width(range = c(1,4)) + geom_node_point(aes(size = centrality, color = group), show.legend = F) + scale_size(range = c(5, 10)) + geom_node_text(aes(label = name), repel = T, size = 5, family="AppleGothic") + theme_graph()
