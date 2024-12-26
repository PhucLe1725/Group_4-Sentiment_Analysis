# Sentiment Analysis
__Nhập môn Học máy và Khai phá dữ liệu__ \
Thành viên: 
- Nguyễn Hữu Huy 
- Nguyễn Nam 
- Lê Xuân Phúc 
- Nguyễn Linh Sơn 
- Phạm Văn Thanh 

## Đề tài 
Phân loại cảm xúc từ văn bản 

## Giới thiệu 
Báo cáo này trình bày nghiên cứu của nhóm về phân tích cảm xúc từ bình luận về phim, sử dụng nhiều mô hình học máy được huấn luyện trên dữ liệu được thu thập tỉ mỉ từ các nguồn trực tuyến. Mục tiêu của nghiên cứu là so sánh hiệu quả giữa các mô hình học máy truyền thống và các mô hình học sâu, nhằm định hướng các giải pháp tối ưu trong thực tế. Chúng tôi đã sử dụng một lượng lớn dữ liệu liên quan đến bình luận phim, bao gồm các thông tin quan trọng như văn bản bình luận và nhãn cảm xúc (tích cực, tiêu cực). Để đảm bảo độ chính xác và độ tin cậy cao trong phân loại cảm xúc, chúng tôi đã triển khai và tối ưu hóa các mô hình học máy như Naive Bayes, Random Forest, SVM,$\ldots$ cũng như các mô hình học sâu tiên tiến như MLP và LSTM.

## Dataset 
[IMDB Movie recommendations](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) \

Bộ dữ liệu gồm 50.000 đánh giá phim từ trang IMDB, với nhãn cảm xúc được phân loại thành tích cực (positive) và tiêu cực (negative). Dữ liệu được cung cấp miễn phí trên trang web của Cornellvà được thu thập từ các bài đánh giá phim.

## Sentence Vectorization
### TF-IDF 
TF-IDF là một kỹ thuật chuyển đổi văn bản thành các vector số dựa trên tần suất xuất hiện của từ trong văn bản và tầm quan trọng của từ trong toàn bộ tập dữ liệu. Mục tiêu là đánh trọng số cao cho các từ quan trọng trong văn bản và giảm trọng số cho các từ phổ biến nhưng ít giá trị thông tin như "not", "movi", "film", ...
$$
\begin{align}
\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)
\end{align}
$$
Trong đó:

TF đo lường mức độ thường xuyên một thuật ngữ xuất hiện trong một tài liệu. Từ xuất hiện nhiều lần trong một văn bản sẽ có TF cao hơn. TF được tính bằng số lần một từ xuất hiện trong một tài liệu chia cho tổng số từ trong tài liệu đó.
$$
\begin{align}
\text{TF}(t, d) = \frac{\text{f}(t, d)}{N}
\end{align}
$$
Với:
- $t$: Từ cần tính TF-IDF
-  $d$: Văn bản hiện tại
- $\text{f}(t, d)$: Số lần từ $t$ xuất hiện trong văn bản $d$
- $N$: Tổng số từ trong văn bản $d$

### Word2Vec
Word2Vec là một kỹ thuật học sâu được sử dụng để biểu diễn các từ trong không gian vector sao cho ngữ nghĩa của từ có thể được ghi nhớ trong các vector này. Các từ có nghĩa tương tự sẽ được biểu diễn bởi các vector tương ứng có hướng gần nhau trong không gian. Word2Vec học vector từ dựa trên ngữ cảnh của từ trong văn bản thông qua hai mô hình: CBOW (Continuous Bag of Words) và Skip-Gram. 

## Các phương pháp truyền thống 

### Naive Bayes
__Thuật toán Naïve Bayes__ là một kỹ thuật phân loại phổ biến dựa trên Định lý Bayes. Nó xem xét các đặc trưng của dữ liệu đầu vào, với giả định rằng các đặc trưng này xuất hiện độc lập với nhau, để dự đoán lớp mà một đối tượng thuộc về. Mặc dù giả định này thường không đúng trong các tình huống thực tế, bộ phân loại Naïve Bayes vẫn được sử dụng rộng rãi nhờ vào độ chính xác cao và độ phức tạp tính toán thấp. 
### k-Nearest Neighbor
__k-Nearest Neighbors__ là một thuật toán học máy giám sát đơn giản nhưng hiệu quả, được sử dụng cho cả nhiệm vụ phân loại và hồi quy. Đây là một thuật toán học không tham số và dựa trên các ví dụ, phân loại các điểm dữ liệu mới dựa trên phiếu bầu đa số hoặc giá trị trung bình của các lớp hoặc giá trị của $k$ láng giềng gần nhất.

### Support Vector Machine
__Support Vector Machine (SVM)__ là một trong những phương pháp học máy giám sát phổ biến, được sử dụng cho bài toán phân loại và hồi quy. Đặc biệt, SVM hiệu quả trong việc xử lý các bài toán phân loại nhị phân và dữ liệu có số lượng mẫu nhỏ hoặc có không gian đặc trưng cao.

### Decision Tree
__Decision Tree Classifier__ là một thuật toán học máy giám sát, sử dụng cây quyết định để phân loại các đối tượng trong một không gian đặc trưng. Mô hình này có thể xử lý các biến đầu vào liên tục và phân loại chúng thành các lớp khác nhau bằng cách chia nhỏ không gian đặc trưng theo các tiêu chí nhất định.
### Random Forest 
__Random Forest__ là một thuật toán học máy mạnh mẽ, được phát triển dựa trên mô hình __Decision Tree__. Random Forest sử dụng một tập hợp các cây quyết định để đưa ra kết quả dự đoán chính xác hơn. Các cây quyết định này được huấn luyện bằng cách sử dụng các mẫu dữ liệu khác nhau, giúp giảm sự phụ thuộc vào bất kỳ cây quyết định cụ thể nào và tăng độ chính xác của mô hình tổng thể.
### Logistic Regression
__Logistic Regression__ là một kỹ thuật học máy được sử dụng rộng rãi để giải quyết các bài toán phân loại, bao gồm cả nhận diện cảm xúc từ bình luận. Mô hình dựa trên cơ sở về phân phối logistic để dự đoán xác suất nhãn của một điểm dữ liệu trong bài toán phân loại nhị phân.

## Các phương pháp học sâu
### Multi-Layers Perceptron
__Multi-Layer Perceptron (MLP)__ là một mạng nơ-ron truyền thẳng (_feed-forward neural network_) bao gồm ba thành phần chính: lớp đầu vào, lớp ẩn và lớp đầu ra. Trong bài toán phân tích cảm xúc (_sentiment analysis_), lớp đầu vào nằm ở phía bên trái, bao gồm các nơ-ron biểu diễn các đặc trưng của văn bản, chẳng hạn như từ hoặc câu đã được mã hóa thành vector (TF-IDF, Word2Vec). Tại lớp ẩn, mỗi nơ-ron thực hiện phép tính tổng trọng số tuyến tính của các đầu vào, biểu diễn bởi công thức:
$$
w_1 x_1 + w_2 x_2 + \dots + w_n x_n
$$
Sau đó, kết quả được đưa qua một hàm kích hoạt phi tuyến, chẳng hạn hàm Sigmoid, ReLU, để mạng có thể học được các mối quan hệ phức tạp giữa các đặc trưng đầu vào và nhãn đầu ra.

Lớp đầu ra của MLP bao gồm một nơ-ron có giá trị thuộc khoảng $\left(0, 1\right)$. Nếu giá trị này lớn hơn $0.5$, mạng dự đoán cảm xúc là tích cực (_positive_); ngược lại, nếu nhỏ hơn hoặc bằng $0.5$, cảm xúc được xác định là tiêu cực (_negative_). Nơ-ron này tổng hợp thông tin từ lớp ẩn cuối cùng để đưa ra kết quả dự đoán cuối cùng.  Muốn đầu ra nằm trong khoảng từ 0 đến 1, nhóm chọn cách đơn giản nhưng hiệu quả là áp dụng hàm kích hoạt sigmoid cho nút đầu ra trước khi đưa ra dự đoán cuối cùng.

$$
 f(x) = \frac{1}{1 + e^{-x}}
 $$
### Long Short-Term Memory (LSTM)
__LSTM__ là một loại mạng nơ-ron hồi tiếp (RNN) được thiết kế để xử lý và dự đoán dữ liệu tuần tự, như chuỗi văn bản hoặc chuỗi thời gian. Khác với RNN truyền thống, LSTM có khả năng lưu giữ thông tin trong thời gian dài, giúp giảm thiểu vấn đề vanishing gradient.

Mỗi nhân LSTM, với cấu trúc gồm các cổng quên (Forget Gate), cổng vào (Input Gate) và cổng xuất (Output Gate), hoạt động như một bộ điều khiển thông tin. Cell state, đóng vai trò như bộ nhớ dài hạn, được cập nhật liên tục dựa trên quyết định của các cổng. Trong khi đó Hidden state truyền tải các thông tin ngắn hạn và giúp tạo ra đầu ra cho mỗi bước thời gian trong chuỗi. Cổng quên quyết định loại bỏ thông tin nào không còn cần thiết, cổng vào quyết định thông tin nào sẽ được ghi vào trạng thái tế bào, và cuối cùng, cổng xuất quyết định phần nào của cell state xuất ra hidden state và sẽ được truyền đến các nhân LSTM tiếp theo.

## Kết quả và đánh giá

__Bảng thống kê các kết quả tốt nhất của từng mô hình__
| Model | Sentence Vectorization | Test Acc. |
| ----- | --------------- | --------- |
| Multinomial Naive Bayes | BoW | 86% |
| k-Nearest Neighbor | Word2Vec | 84.78% |
| Decision Tree | Word2Vec | 75% |
| Random Forest | Word2Vec | 86.5% |
| SVM | Word2Vec | 89.11% |
| Logistic Regression | TF-IDF | 89.3% |
| MLP | Word2Vec | 89.02% |
| BiLSTM | Word2Vec | 91.2% |

_Chú ý: các kết quả trong `src/python` có thể có khác biệt nhỏ với kết quả trên đây, nguyên nhân tới từ sự bất đồng bộ trong khâu xử ký dữ liệu ban đầu, ngoài ra quá trình huấn luyện một số mô hình có thể bị ảnh hưởng bới tính ngẫu nhiên_

Trong nghiên cứu này, chúng tôi đã xây dựng và triển khai nhiều mô hình khác nhau để phân loại cảm xúc trên bộ dữ liệu IMDB. Qua quá trình tiền xử lý, vector hóa và huấn luyện, nhóm nghiên cứu rút ra những kết luận quan trọng như sau:


- __Ưu điểm của Word Embeddings:__ Các phương pháp Word Embeddings, đặc biệt là Word2Vec, đa số thể hiện hiệu quả vượt trội so với kỹ thuật Bag-of-Words (BoW), TF-IDF trong việc nắm bắt ngữ nghĩa và mối quan hệ giữa các từ. Word2Vec cho phép biểu diễn các từ trong không gian vector liên tục, giúp cải thiện đáng kể độ chính xác của các mô hình phân loại cảm xúc, đặc biệt là các mô hình dựa trên mạng nơ-ron như BiLSTM.

-  __Các mô hình truyền thống:__ Các mô hình truyền thống, bên cạnh điểm mạnh về khả năng giải thích được (_explanable_), cũng đã cho thấy hiệu suất tương đối tốt cả về độ chính xác trên tập kiểm tra, đặc biệt là ở Logistic Regression, Random Forest và SVM. Điều này chứng tỏ vai trò của các mô hình truyền thống vẫn rất đáng để quan tâm.
    
-  __Hiệu quả của BiLSTM:__ Trong số các mô hình được nghiên cứu, BiLSTM cho thấy hiệu suất tốt nhất trong việc phân loại cảm xúc, đạt độ chính xác cao trên tập kiểm tra. Khả năng phân tích ngữ cảnh hai chiều từ chuỗi văn bản, kết hợp với Word2Vec, đã giúp BiLSTM đạt được kết quả vượt trội.
    
