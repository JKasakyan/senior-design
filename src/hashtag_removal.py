import json

def removeHashtags(tweet):
    text = tweet["text"].split()
    ind = 0
    inds = []
    word_count = len(text)
    for word in text:
        if word[0] == "#":
            inds.append(ind)
        ind += 1
    if len(inds) == 1:
        if text[inds[0]].lower() != "#sarcasm":
            text[inds[0]] = text[inds[0]][1:]
    else:
        if len(inds)>0 and inds[0] == 0:
            i = 1
            removed = False
            while i < len(inds) and inds[i] == inds[i-1]+1:
                if text[inds[i-1]].lower() != "#sarcasm":
                    text[inds[i-1]] = ""
                i += 1
                removed = True
            if removed:
                if text[inds[i-1]].lower() != "#sarcasm":
                    text[inds[i-1]] = ""
        if len(inds)>0 and inds[len(inds)-1] == word_count-1:
            i = len(inds)-1
            removed = False
            while i >= 0 and inds[i] == inds[i-1]+1:
                if text[inds[i]].lower() != "#sarcasm":
                    text[inds[i]] = ""
                i -= 1
                removed = True
            if removed:
                if text[inds[i]].lower() != "#sarcasm":
                    text[inds[i]] = ""
        for ind in inds:
            if text[ind] != "":
                if text[ind].lower() != "#sarcasm":
                    text[ind] = text[ind][1:]
    i = 0
    while i < len(text):
        if text[i] == "":
            del text[i]
        else:
            i += 1
    tw = {}
    tw["id"] = tweet["id"]
    tw["text"] = " ".join(text)
    tw["media"] = False
    tw["test"] = True
    tw["urls"] = False
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
