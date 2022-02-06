from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

emails = fetch_20newsgroups(categories=['rec.sport.baseball', 'rec.sport.hockey'])
print(emails.target_names)

print(emails.data[5])
print(emails.target[5])

train_emails = fetch_20newsgroups(categories=['comp.sys.ibm.pc.hardware','rec.sport.hockey', 'comp.sys.ibm.pc.hardware','sci.crypt'], subset = 'train', shuffle = True, random_state = 108)

test_emails = fetch_20newsgroups(categories=['comp.sys.ibm.pc.hardware','rec.sport.hockey', 'comp.sys.ibm.pc.hardware', 'sci.crypt'], subset = 'test', shuffle = True, random_state = 108)

counter = CountVectorizer()
counter.fit(test_emails.data + train_emails.data)

train_counts = counter.transform(train_emails.data)
#print(train_counts)
test_counts = counter.transform(test_emails.data)

classifier = MultinomialNB()
classifier.fit(train_counts, train_emails.target)

print(classifier.score(test_counts, test_emails.target))