import tensorflow as tf
import numpy as np

from tensorflow.contrib import rnn

import pickle

tf.set_random_seed(777)  # reproducibility
np.set_printoptions(suppress=True)


class Model2:
    def __init__(self, train_data, test_data, char_dic, model_name='auto_spacing'):
        '''
            init hyper parameter
        '''
        self.epoch = 3
        self.embedding_size = 300
        self.hidden_size = 200
        self.num_classes = 2 # B, I
        self.num_rnn_layer = 3

        self.model_name = model_name
        self.model_dir = './save/' + model_name


        '''
            init data set 
        '''
        self.train_data = train_data
        self.test_data = test_data
        self.char_dic = char_dic

        '''
            init model 
        '''
        self._init_placeholder()
        self._init_variable()
        self._make_graph()

    def _init_placeholder(self):
        # batch, seq
        self.x = tf.placeholder(tf.int32, [None, None])
        self.y = tf.placeholder(tf.int32, [None, None])

        self.len = tf.placeholder(tf.int32, [None])
        self.max_len = tf.placeholder(tf.int32)

        self.batch_size = tf.placeholder(tf.int32, [])

        self._loss = tf.placeholder(tf.float32)
        self._acc = tf.placeholder(tf.float32)

    def _make_bilstm_n_layer(self):
        self.cell_fw = rnn.MultiRNNCell([rnn.LSTMCell(self.hidden_size) for _ in range(self.num_rnn_layer)])
        self.cell_bw = rnn.MultiRNNCell([rnn.LSTMCell(self.hidden_size) for _ in range(self.num_rnn_layer)])
        self.init_state_fw = self.cell_fw.zero_state(self.batch_size, dtype=tf.float32)
        self.init_state_bw = self.cell_bw.zero_state(self.batch_size, dtype=tf.float32)

        self.outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.cell_fw,
                                                              cell_bw=self.cell_bw,
                                                              inputs=self.embedding,
                                                              sequence_length=self.len,
                                                              initial_state_fw=self.init_state_fw,
                                                              initial_state_bw=self.init_state_bw,
                                                              dtype=tf.float32)

        self.outputs = tf.concat(self.outputs, 2)
        self.outputs = tf.contrib.layers.fully_connected(inputs=self.outputs,
                                                         num_outputs=self.num_classes,
                                                         activation_fn=None)

        # logits : batch, seq, hidden
        # targets : batch, seq
        self.weight = tf.ones([self.batch_size, self.max_len])
        self.loss = tf.contrib.seq2seq.sequence_loss(logits=self.outputs,
                                                     targets=self.y,
                                                     weights=self.weight)

    def _make_bilstm_crf_n_layer(self):
        self.cell_fw = rnn.MultiRNNCell([rnn.LSTMCell(self.hidden_size) for _ in range(self.num_rnn_layer)])
        self.cell_bw = rnn.MultiRNNCell([rnn.LSTMCell(self.hidden_size) for _ in range(self.num_rnn_layer)])
        self.init_state_fw = self.cell_fw.zero_state(self.batch_size, dtype=tf.float32)
        self.init_state_bw = self.cell_bw.zero_state(self.batch_size, dtype=tf.float32)

        self.outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.cell_fw,
                                                              cell_bw=self.cell_bw,
                                                              inputs=self.embedding,
                                                              sequence_length=self.len,
                                                              initial_state_fw=self.init_state_fw,
                                                              initial_state_bw=self.init_state_bw,
                                                              dtype=tf.float32)

        self.outputs = tf.concat(self.outputs, 2)
        self.outputs = tf.contrib.layers.fully_connected(inputs=self.outputs,
                                                         num_outputs=self.num_classes,
                                                         activation_fn=None)

        self.log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(self.outputs, self.y, self.len)
        self.outputs, viterbi_score = tf.contrib.crf.crf_decode(self.outputs, transition_params, self.len)
        self.loss = tf.reduce_mean(-self.log_likelihood)

    def _init_variable(self):
        self.word_embeddings = tf.get_variable(
            'word_embeddings',
            [len(self.char_dic), self.embedding_size]
        )

    def _make_graph(self):
        self.embedding = tf.nn.embedding_lookup(self.word_embeddings, self.x)

        self._make_bilstm_crf_n_layer()

        self.optimizer = tf.train.AdamOptimizer()
        self.optimize = self.optimizer.minimize(self.loss)

        self.result = tf.equal(self.outputs, self.y)

        self.s1 = tf.summary.scalar('loss', self._loss)
        self.s2 = tf.summary.scalar('acc', self._acc)
        self.merged = tf.summary.merge_all()


    def _padding(self, lst, value=0):
        lengths = [len(elem) for elem in lst]
        max_length = max(lengths)

        for elem in lst:
            for _ in range(max_length - len(elem)):
                elem.append(value)

        return lst, lengths

    def _char2idx(self, x_data):
        return [[self.char_dic[ch] for ch in sentence] for sentence in x_data]

    def _label2idx(self, y_data):
        return [[1 if l == 'B' else 0 for l in label] for label in y_data]

    def _accuracy(self, result, lens):
        return 100. * sum(sum(seq[:l]) for seq, l in zip(result, lens)) / sum(lens)

    def _count_correct(self, result, lens):
        return sum(sum(seq[:l]) for seq, l in zip(result, lens))


    def train(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()

            for e in range(self.epoch):
                for i, (x_data, y_data) in enumerate(self.train_data, 1):
                    x_data = self._char2idx(x_data)
                    y_data = self._label2idx(y_data)

                    x_data, x_len = self._padding(x_data, value=0)
                    y_data, y_len = self._padding(y_data, value=0)

                    feed_dict = {
                        self.x : x_data,
                        self.y : y_data,
                        self.len : x_len,
                        self.max_len : max(x_len),
                        self.batch_size : len(x_data)
                    }
                    _, loss, result, outputs = sess.run(
                        [self.optimize, self.loss, self.result, self.outputs],
                        feed_dict=feed_dict
                    )

                    if i % 10 == 0:
                        print('[cur] epoch : {0}, idx : {1}, loss : {2:.6f}'.format(e, i, loss))
                        print(outputs[0][:x_len[0]])
                        save_path = saver.save(sess, self.model_dir)

                    # if i % 10 is 0:
                    #     '''
                    #         add result to tensor board
                    #     '''
                    #     acc = self._test(sess)
                    #
                    #     if acc > max_result['acc']:
                    #         max_result['e'] = e
                    #         max_result['i'] = i
                    #         max_result['loss'] = loss
                    #         max_result['acc'] = acc
                    #
                    #     # feed_dict2 = {self._loss: loss,
                    #     #               self._acc: acc}
                    #     #
                    #     # summary = sess.run(self.merged, feed_dict=feed_dict2)
                    #     # writer.add_summary(summary, step)
                    #     # step += 1
                    #
                    #     # save_path = saver.save(sess, self.model_dir + '_{}'.format(e))
                    #         save_path = saver.save(sess, self.model_dir)
                    #
                    #     print('[cur] epoch : {0}, idx : {1}, loss : {2:.6f}, acc : {3:.6f}'.format(e, i, loss, acc))
                    #     print('[max] epoch : {0}, idx : {1}, loss : {2:.6f}, max_acc : {3:.6f}'.format(max_result['e'], max_result['i'], max_result['loss'], max_result['acc']))
                    #     print(outputs[0][:x_len[0]])


    def _test(self, sess):
        cnt = 0
        total_cnt = 0

        for i, (x_data, y_data) in enumerate(self.test_data, 1):
            x_data = self._char2idx(x_data)
            y_data = self._label2idx(y_data)

            x_data, x_len = self._padding(x_data, value=0)
            y_data, y_len = self._padding(y_data, value=0)

            feed_dict = {self.x : x_data,
                         self.y : y_data,
                         self.len : x_len,
                         self.max_len : max(x_len),
                         self.batch_size : len(x_data),
                         self.keep_prob : 1.0 }

            result = sess.run(self.result, feed_dict=feed_dict)

            cnt += self._count_correct(result, x_len)
            total_cnt += sum(x_len)

        return cnt / total_cnt



    def test(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()

            saver.restore(sess, self.model_dir)

            cnt = 0
            total_cnt = 0

            ofs = open('pred.txt', 'w')

            for i, (x_data, y_data) in enumerate(self.test_data, 1):
                print(i, len(x_data))
                x_data = self._char2idx(x_data)
                y_data = self._label2idx(y_data)

                x_data, x_len = self._padding(x_data, value=0)
                y_data, y_len = self._padding(y_data, value=0)

                feed_dict = {self.x : x_data,
                             self.y : y_data,
                             self.len : x_len,
                             self.max_len : max(x_len),
                             self.batch_size : len(x_data)}

                result = sess.run(self.outputs, feed_dict=feed_dict)

                # cnt += self._count_correct(result, x_len)
                # total_cnt += sum(x_len)

                # print(x_data)
                # print(y_data)


                for seq, l in zip(result, x_len):
                    ofs.write(''.join('B' if tag else 'I' for tag in seq[:l]) + '\n')
                    # print(''.join('B' if tag else 'I' for tag in seq[:l]))

            ofs.close()
                # print('idx : {0}, acc : {1:.6f}'.format(i, 100. * cnt / total_cnt))
            # print('total test acc : {0:.6f}'.format(100. * cnt / total_cnt))