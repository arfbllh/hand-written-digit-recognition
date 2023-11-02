import mnist_loader

from sklearn import svm

def svm_baseline():
    train_data, validation_data, test_data = mnist_loader.load_data()

    clf = svm.SVC()
    print('running')
    clf.fit(train_data[0], train_data[1])
    
    predictions = [int (a) for a in clf.predict(test_data[0])]

    num_correct = sum(int(a == y) for a, y in zip(predictions, test_data[1]))

    print(num_correct)

svm_baseline()