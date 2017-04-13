import json
try:
    import HTMLParser
    parser = HTMLParser.HTMLParser()
except:
    import html
    parser = html
    
def removeHashtags(tweet):
    text = tweet["text"].split()
    ind = 0
    inds = []
    tags_list = []
    word_count = len(text)
    for word in text:
        if word[0] == "#":
            inds.append(ind)
        ind += 1
    if len(inds) == 1:
        tags_list.append(text[inds[0]][1:])
        if text[inds[0]].lower() != "#sarcasm":
            text[inds[0]] = text[inds[0]][1:]
        else:
            text[inds[0]] = ""
    else:
        if len(inds)>0 and inds[0] == 0:
            i = 1
            removed = False
            while i < len(inds) and inds[i] == inds[i-1]+1:
                tags_list.append(text[inds[i-1]][1:])
                text[inds[i-1]] = ""
                i += 1
                removed = True
            if removed:
                tags_list.append(text[inds[i-1]][1:])
                text[inds[i-1]] = ""
        if len(inds)>0 and inds[len(inds)-1] == word_count-1:
            i = len(inds)-1
            removed = False
            while i >= 0 and inds[i] == inds[i-1]+1:
                tags_list.append(text[inds[i]][1:])
                text[inds[i]] = ""
                i -= 1
                removed = True
            if removed:
                tags_list.append(text[ind][1:])
                text[inds[i]] = ""
        for ind in inds:
            if text[ind] != "":
                if text[ind].lower() != "#sarcasm":
                    tags_list.append(text[ind][1:])
                    text[ind] = text[ind][1:]
                else:
                    tags_list.append(text[ind][1:])
                    text[ind] = ""
    i = 0
    while i < len(text):
        if text[i] == "":
            del text[i]
        else:
            i += 1
    tw = {"id": tweet["id"], "text": " ".join(text), "media": tweet["media"], "urls": tweet["urls"], "tags": tags_list}
    return tw

def replaceHtml(tw):
    tw["text"] = parser.unescape(tw["text"])
    return tw

def processTweets(tweets):
    my_tw = []
    for tweet in tweets:
        tw = removeHashtags(tweet)
        my_tw.append(tw)
    outfile = open("out_tweet.json","w")
    json.dump(my_tw,outfile,indent=1)

if __name__ == "__main__":
    tw_file = open("tw.json","r")
    tweets = json.load(tw_file)
    processTweets(tweets)
