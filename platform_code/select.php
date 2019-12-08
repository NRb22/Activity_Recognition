
<?php
    DEFINE('DB_USERNAME', 'root');
    DEFINE('DB_PASSWORD', '');
    DEFINE('DB_HOST', 'localhost');
    DEFINE('DB_DATABASE', 'photo');
    
    $mysqli = new mysqli(DB_HOST, DB_USERNAME, DB_PASSWORD, DB_DATABASE);
    
    if (mysqli_connect_error()) {
        die('Connect Error ('.mysqli_connect_errno().') '.mysqli_connect_error());
    }
    
    echo 'Connected successfully.';
    $sql="SELECT * FROM PHOTO";
    $result=mysqli_query($mysqli,$sql);
    $row=mysqli_fetch_array($result);
    $src=$row['name'];
    echo "<img src='$src'>";
    
    $mysqli->close();
?>


if (isset($_POST['userid']) && isset($_POST['password'])){
    //mysql select 쿼리로 userid, password가 일치하는 userid 값 가져오기
    $query = "select * from security where userid='$userid' and password = password('$password') ";
    $result = mysql_query($query);
    $array = mysql_fetch_array($result);
    
    
    if(!empty($array['userid'])){
        $_SESSION['userid'] = $array['userid'];
        echo("<script>alert(\"LOGIN SUCCESS\");location.href='./main.php';</script>");
    }
    else{
        echo("<script>alert(\"ID 혹은 PW가 잘못되었습니다.\");</script>");
        ?><script>history.back()</script><?
        
        
    }
    
    
}else{
    echo("<script>alert(\"로그인 후 이용하세요\");</script>");
    echo("<script>location.href='./index.php';</script>");
}

//안되어있다면 index.php 로 돌아가기(else)


?>
