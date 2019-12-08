<?php
  header('Content-Type: text/html; charset=utf-8');
?>
<?php session_start();
    DEFINE('DB_USERNAME', 'root');
    DEFINE('DB_PASSWORD', '');
    DEFINE('DB_HOST', 'localhost');
    DEFINE('DB_DATABASE', 'LearningEyes');

    $mysqli = new mysqli(DB_HOST, DB_USERNAME, DB_PASSWORD, DB_DATABASE);

    if (mysqli_connect_error()) {
        die('Connect Error ('.mysqli_connect_errno().') '.mysqli_connect_error());
    }
   	//Index.php에서 받은 userid, password가 설정되어있는지 확인 (if)
    $userid = $_POST['userid'];
    $password = $_POST['password'];
    
    $sql="select * from employee where userid='$userid'";
    $result=mysqli_query($mysqli,$sql);
    $row=mysqli_fetch_array($result);
    $hash_pw=$row['password'];

    //Index.php에서 받은 userid, password가 설정되어있는지 확인 (if)
    if($userid==$row['userid']&&password_verify($password,$hash_pw)){
        $_SESSION['userid']=$row['userid'];
        $_SESSION['password']=$password;
        echo "<script>location.href='main.php';</script>";
    }
    else{
        echo"<script>alert('check ID or PW');history.back();</script>";
    }
    ?>
