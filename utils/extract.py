import ast
import pandas as pd


def extract_review_activity(df):
    cols=['acc_num','review_id','sortTimestamp','rating','helpfulVotes','title','text','writtenInLanguageCode','images_posted','asin','verifiedPurchase']
    
    extracted_data = []
    for index, row in df.iterrows():
        print("Extracting {} out of {}...".format(index+1,len(df)))
        not_empty_data = True
        try:
            contribution_data = ast.literal_eval(row['reviewer_contributions'])
            if contribution_data == []:
                not_empty_data = False
        except:
            not_empty_data = False
        if not_empty_data:
            count = 0
            for data in contribution_data:
                data = dict(data)
                if 'ideas' not in data['id']:
                    try:
                        acc_num = row['acc_num']
                        review_id = data['externalId']
                        sortTimestamp = data['sortTimestamp']
                        text = data['text']
                        asin = data['product']['asin']
                        verifiedPurchase = data['verifiedPurchase']
                        writtenInLanguageCode = data['writtenInLanguageCode']

                        if 'rating' not in data:
                            rating = None
                        else:
                            rating = data['rating']
                        if 'helpfulVotes' not in data:
                            helpfulVotes = None
                        else:
                            helpfulVotes = data['helpfulVotes']
                        if 'title' not in data:
                            title = ''
                        else:
                            title = data['title']
                        if 'images' not in data:
                            images_posted = None
                        else:
                            images_posted = len(data['images'])

                        extracted_data.append([acc_num,review_id,sortTimestamp,rating,helpfulVotes,title,text,writtenInLanguageCode,images_posted,asin,verifiedPurchase])
                        count += 1
                        print("Added {} out of {} from reviewer contribution...".format(count,len(data)))
                    except:
                        continue

            print("Reviewer Activity Dataset currently has {} rows...\n".format(len(extracted_data)))

    review_activity_df = pd.DataFrame(extracted_data,columns=cols)
    return review_activity_df.drop_duplicates(subset='review_id')