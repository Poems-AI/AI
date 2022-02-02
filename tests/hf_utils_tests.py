from poemsai.hf_utils import model_to_url


def test_model_to_url():
    url_no_pwd = model_to_url('t5', 'user1')
    url_pwd = model_to_url('t5', 'user1', pwd='mypass')

    assert url_no_pwd == 'https://huggingface.co/user1/t5'
    assert url_pwd == 'https://user1:mypass@huggingface.co/user1/t5'
